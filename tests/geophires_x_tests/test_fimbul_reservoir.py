from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np
import CoolProp.CoolProp as CP

from tests.base_test_case import BaseTestCase

# Ruff disabled because imports are order-dependent
# ruff: noqa: I001
from geophires_x.Model import Model
from geophires_x.FimbulReservoir import FimbulReservoir, FIMBUL_PROFILES_DIR_ENV_VAR, FLUID_SYSTEM_H2O
from geophires_x.CoupledWellBores import CoupledWellBores
from geophires_x.OptionList import RedrillingTriggerTemperature
from geophires_x.OptionList import ReservoirModel
from geophires_x_client import GeophiresInputParameters
from geophires_x_client import GeophiresXClient
from geophires_x_client import GeophiresXResult

SUPERHOT_WELLBORE_AVAILABLE = importlib.util.find_spec('superhot_wellbore') is not None

_EXAMPLE_FILE = 'examples/example_FIMBUL-1.txt'
_GRINGARTEN_EXAMPLE_FILE = 'examples/example_SHR-6.txt'
_FIXTURE_FILE = 'examples/fimbul_profiles/example_FIMBUL-1.json'

_TIMESTEPS_PER_YEAR = 12

_COUPLED_WELLBORE_INPUT_PREFIXES = ('Production Wellbore Model', 'Coupled Wellbore', 'Power Plant Type')
"""Input lines dropped from the example to run it with the standard (Ramey) wellbore model and a conventional plant"""


def _test_file_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / name


def _without_inputs(example_path: str, without: tuple[str, ...] = (), without_prefixes: tuple[str, ...] = ()) -> str:
    """A copy of the example input file without the named inputs (and the inputs starting with the given prefixes)."""
    with open(example_path, encoding='utf-8') as f:
        lines = [
            line for line in f if line.split(',')[0].strip() not in without and not line.startswith(without_prefixes)
        ]
    path = os.path.join(tempfile.gettempdir(), f'example_FIMBUL-{uuid.uuid4()!s}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return path


def _read_model(example_path: str, params: dict | None = None, without: tuple[str, ...] = ()) -> Model:
    """Build and read (but not calculate) a model from an example input file with optional overrides."""
    if without:
        example_path = _without_inputs(example_path, without)

    input_params = GeophiresInputParameters(from_file_path=example_path, params=params)
    stash_cwd = Path.cwd()
    stash_sys_argv = sys.argv
    sys.argv = ['', input_params.as_file_path()]
    try:
        model = Model(enable_geophires_logging_config=False)
        model.read_parameters()
    finally:
        sys.argv = stash_sys_argv
        os.chdir(stash_cwd)
    return model


def build_gringarten_shaped_feedzone_history(
    model: Model, timesteps_per_year: int = _TIMESTEPS_PER_YEAR, horizon_years: float | None = None
) -> dict:
    """
    A FeedzoneHistory JSON document (schema_version 1, the fimbul-python hand-off format) shaped like the Gringarten
    reservoir output temperature of the given model (whose reservoir has been calculated but not redrilled), so that
    the Fimbul reservoir model can be exercised without Julia. The temperature series is the Gringarten curve
    resampled onto t = 0, 1/12, ... yr; pressure and enthalpy are those of the feedzone liquid after the
    productivity-index drawdown; the mass rate is the prescribed flow rate; the Gringarten block is the curve itself.
    """
    reserv = model.reserv
    wellbores = model.wellbores
    grid_yr = np.array(reserv.timevector.value, dtype=float)
    Tres_C = np.array(reserv.Tresoutput.value, dtype=float)
    if horizon_years is None:
        horizon_years = float(model.surfaceplant.plant_lifetime.value)
    time_yr = np.arange(0, round(horizon_years * timesteps_per_year) + 1) / timesteps_per_year
    T_C = np.interp(time_yr, grid_yr, Tres_C)

    flow_kgs = float(wellbores.prodwellflowrate.value)
    P_reservoir_kPa = float(wellbores._initial_reservoir_pressure_kPa(model))
    PI_kg_per_s_per_kPa = float(wellbores.PI.quantity().to('kg/s/bar').magnitude) / 100.0
    P_fz_MPa = (P_reservoir_kPa - flow_kgs / PI_kg_per_s_per_kPa) / 1000.0
    P_MPa = np.full(len(time_yr), P_fz_MPa)
    h_MJkg = np.array([CP.PropsSI('H', 'T', T + 273.15, 'P', P_fz_MPa * 1e6, 'Water') / 1e6 for T in T_C])
    P_sat_MPa = np.array([CP.PropsSI('P', 'T', T + 273.15, 'Q', 0, 'Water') / 1e6 for T in T_C])

    depth_km = float(reserv.depth.quantity().to('km').magnitude)
    return {
        'schema_version': '1',
        'source': 'geophires-gringarten-synthetic',
        'scenario': {
            'dcm': 'synthetic',
            'target_temperature_C': float(reserv.Trock.value),
            'gradient_C_per_km': float(reserv.averagegradient.value) * 1000.0,
            'reservoir_depth_km': depth_km,
            'flow_rate_per_well_kg_s': flow_kgs,
            'fracture_height_m': float(reserv.fracheightcalc.value),
            'injection_temperature_C': float(wellbores.Tinj.value),
            'surface_temperature_C': float(reserv.Tsurf.value),
            'plant_lifetime_years': int(model.surfaceplant.plant_lifetime.value),
            'horizon_years': float(horizon_years),
        },
        'time_yr': [float(t) for t in time_yr],
        'T_feedzone_C': [float(T) for T in T_C],
        'P_feedzone_MPa': [float(P) for P in P_MPa],
        'h_feedzone_MJkg': [float(h) for h in h_MJkg],
        'mass_rate_kgs': [flow_kgs] * len(time_yr),
        'gringarten': {'Tresoutput_C': [float(T) for T in Tres_C], 'mean_C': float(np.mean(Tres_C))},
        'invariants': {
            'min_saturation_margin_MPa': float(np.min(P_MPa - P_sat_MPa)),
            'rate_drift_fraction': 0.0,
            'energy_balance_ratio': 1.0,
            'horizon_ok': True,
        },
        'provenance': {
            'case_hash': 'synthetic-gringarten-example_SHR-6',
            'namespace': 'synthetic',
            'note': 'Synthetic Gringarten-shaped profile generated from example_SHR-6 by '
            'tests/geophires_x_tests/test_fimbul_reservoir.py; not a Fimbul run.',
        },
    }


def regenerate_example_fixture() -> Path:
    """Write tests/examples/fimbul_profiles/example_FIMBUL-1.json from example_SHR-6's Gringarten curve."""
    model = _read_model(str(_test_file_path(_GRINGARTEN_EXAMPLE_FILE)))
    model.reserv.Calculate(model)
    history = build_gringarten_shaped_feedzone_history(model)
    path = _test_file_path(_FIXTURE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=1)
        f.write('\n')
    return path


class FimbulReservoirTestCase(BaseTestCase):
    """
    Tests of the Fimbul feedzone profile reservoir model (Reservoir Model 9). None of them needs Julia or fimbul; the
    profiles are a committed synthetic document (single-phase liquid) and small synthetic pressure-enthalpy water (H2O)
    documents written to a temporary directory. The ones that run the coupled inflow-wellbore production wellbore model
    need the optional superhot-wellbore package and are skipped without it.
    """

    def _example_path(self) -> str:
        return self._get_test_file_path(Path('..', _EXAMPLE_FILE))

    def _fixture(self) -> dict:
        with open(_test_file_path(_FIXTURE_FILE), encoding='utf-8') as f:
            return json.load(f)

    def _write_profile(self, history: dict, directory: str | None = None) -> str:
        path = os.path.join(directory or tempfile.gettempdir(), f'fimbul-profile-{uuid.uuid4()!s}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f)
        return path

    def _model(self, params: dict | None = None, without: tuple[str, ...] = ()) -> Model:
        return _read_model(self._example_path(), params=params, without=without)

    def _reservoir_model(self, history: dict | None = None, params: dict | None = None) -> Model:
        """A read model whose reservoir (only) has been calculated, from the fixture or the given profile."""
        overrides = dict(params or {})
        if history is not None:
            overrides['Fimbul Feedzone Profile File'] = self._write_profile(history)
        model = self._model(overrides)
        model.reserv.Calculate(model)
        return model

    def test_reservoir_model_selection(self):
        model = self._model()
        self.assertIsInstance(model.reserv, FimbulReservoir)
        self.assertEqual(model.reserv.resoption.value, ReservoirModel.FIMBUL)
        self.assertEqual(model.reserv.resoption.value.display_name, 'Fimbul Feedzone Profile Model')

    def test_profile_file_is_required(self):
        with self.assertRaises(ValueError) as cm:
            self._model(without=('Fimbul Feedzone Profile File',))
        self.assertIn('Fimbul Feedzone Profile File', str(cm.exception))

    def test_missing_profile_file_is_reported(self):
        model = self._model({'Fimbul Feedzone Profile File': 'does-not-exist.json'})
        with self.assertRaises(FileNotFoundError) as cm:
            model.reserv.Calculate(model)
        self.assertIn(FIMBUL_PROFILES_DIR_ENV_VAR, str(cm.exception))

    def test_profile_resolved_relative_to_profiles_dir(self):
        stash = os.environ.get(FIMBUL_PROFILES_DIR_ENV_VAR)
        os.environ[FIMBUL_PROFILES_DIR_ENV_VAR] = str(_test_file_path('examples/fimbul_profiles'))
        try:
            path = FimbulReservoir.resolve_profile_path('example_FIMBUL-1.json')
        finally:
            if stash is None:
                del os.environ[FIMBUL_PROFILES_DIR_ENV_VAR]
            else:
                os.environ[FIMBUL_PROFILES_DIR_ENV_VAR] = stash
        self.assertEqual(path, _test_file_path(_FIXTURE_FILE).resolve())

    def test_example_fixture_matches_generator(self):
        """The committed synthetic profile is what the generator produces from example_SHR-6's Gringarten curve."""
        fixture = self._fixture()
        model = _read_model(str(_test_file_path(_GRINGARTEN_EXAMPLE_FILE)))
        model.reserv.Calculate(model)
        generated = build_gringarten_shaped_feedzone_history(model)

        self.assertEqual(fixture['schema_version'], '1')
        self.assertEqual(fixture['scenario'], generated['scenario'])
        np.testing.assert_allclose(fixture['time_yr'], generated['time_yr'], atol=1e-9)
        np.testing.assert_allclose(fixture['T_feedzone_C'], generated['T_feedzone_C'], atol=1e-6)
        np.testing.assert_allclose(
            fixture['gringarten']['Tresoutput_C'], generated['gringarten']['Tresoutput_C'], atol=1e-6
        )
        self.assertEqual(len(fixture['time_yr']), 25 * _TIMESTEPS_PER_YEAR + 1)
        self.assertEqual(fixture['time_yr'][0], 0.0)

    def test_profile_sets_reservoir_output_temperature(self):
        fixture = self._fixture()
        model = self._reservoir_model()
        reserv = model.reserv

        self.assertEqual(len(reserv.Tresoutput.value), len(reserv.timevector.value))
        self.assertAlmostEqual(reserv.Tresoutput.value[0], fixture['T_feedzone_C'][0], delta=1e-9)
        self.assertAlmostEqual(reserv.Tresoutput.value[0], reserv.Trock.value, delta=0.5)
        self.assertLess(reserv.Tresoutput.value[-1], reserv.Tresoutput.value[0])
        # resampling of a Gringarten-shaped profile onto the Gringarten grid is nearly the identity
        np.testing.assert_allclose(reserv.Tresoutput.value, fixture['gringarten']['Tresoutput_C'], atol=0.05)

        self.assertTrue(reserv.provides_feedzone_history)
        self.assertEqual(reserv.drawdown_reference_steps, 12)
        self.assertEqual(reserv.max_redrill_events, 12)
        self.assertEqual(
            reserv.tiled_histories,
            ('fimbul_feedzone_pressure_kPa', 'fimbul_feedzone_enthalpy_kJ_kg', 'fimbul_mass_rate_kg_s'),
        )
        for name in reserv.tiled_histories:
            self.assertEqual(len(getattr(reserv, name)), len(reserv.timevector.value))
        self.assertAlmostEqual(
            reserv.fimbul_feedzone_pressure_kPa[0], fixture['P_feedzone_MPa'][0] * 1000.0, delta=1e-6
        )
        self.assertAlmostEqual(reserv.fimbul_mass_rate_kg_s[0], 60.0, delta=1e-9)
        self.assertEqual(reserv.fimbul_case_hash, fixture['provenance']['case_hash'])
        self.assertEqual(reserv.fimbul_case_hash_output.value, fixture['provenance']['case_hash'])
        # a single-phase liquid profile (no fluid_system): the saturation margin, not the vapor saturation
        self.assertEqual(reserv.fimbul_fluid_system, '')
        self.assertEqual(reserv.fimbul_fluid_system_output.value, '')
        self.assertGreater(reserv.fimbul_min_saturation_margin.value, 0)
        self.assertTrue(math.isnan(reserv.fimbul_max_vapor_saturation_output.value))
        self.assertAlmostEqual(reserv.fimbul_achieved_flow_rate.value, 60.0, delta=1e-9)
        # Fimbul minus Gringarten: the synthetic profile is the Gringarten curve, so the difference is tiny
        self.assertTrue(math.isfinite(reserv.fimbul_gringarten_mean_delta.value))
        self.assertLess(abs(reserv.fimbul_gringarten_mean_delta.value), 0.05)
        self.assertAlmostEqual(reserv.fimbul_gringarten_mean_delta_C, reserv.fimbul_gringarten_mean_delta.value)

    def test_whole_history_is_overwritten_on_recalculation(self):
        """The base reservoir calculation is cached and redrilling edits the history in place; Calculate restores it."""
        model = self._reservoir_model()
        original = np.array(model.reserv.Tresoutput.value, dtype=float)
        model.reserv.Tresoutput.value = np.tile(original[:10], 30)[: len(original)]
        model.reserv.Calculate(model)
        np.testing.assert_allclose(model.reserv.Tresoutput.value, original)

    def test_scenario_mismatch_raises(self):
        for key, value, label in (
            ('reservoir_depth_km', 5.0, 'Reservoir Depth'),
            ('target_temperature_C', 250.0, 'bottom-hole temperature'),
            ('flow_rate_per_well_kg_s', 80.0, 'Production Flow Rate per Well'),
            ('fracture_height_m', 192.1, 'fracture height'),
        ):
            with self.subTest(key=key):
                history = self._fixture()
                history['scenario'][key] = value
                with self.assertRaises(ValueError) as cm:
                    self._reservoir_model(history)
                self.assertIn(label, str(cm.exception))
                self.assertIn('does not belong to this scenario', str(cm.exception))

    def test_initial_temperature_tolerance(self):
        history = self._fixture()
        Trock = history['scenario']['target_temperature_C']
        history['T_feedzone_C'] = [T + 8.0 for T in history['T_feedzone_C']]  # above 5 C: warning only
        with self.assertLogs(level='WARNING') as logs:
            model = self._reservoir_model(history)
        self.assertTrue(any('differs from the bottom-hole temperature' in it for it in logs.output))
        self.assertAlmostEqual(model.reserv.Tresoutput.value[0], Trock + 8.0, delta=1e-6)

        history['T_feedzone_C'] = [T + 30.0 for T in self._fixture()['T_feedzone_C']]  # above 4 x 5 C: error
        with self.assertRaises(ValueError) as cm:
            self._reservoir_model(history)
        self.assertIn('Fimbul Initial Temperature Tolerance', str(cm.exception))

    def test_invalid_documents_raise(self):
        cases = {
            'schema': lambda h: h.update(schema_version='2'),
            'missing series': lambda h: h.pop('h_feedzone_MJkg'),
            'length': lambda h: h['T_feedzone_C'].pop(),
            'start': lambda h: h.update(time_yr=[t + 0.01 for t in h['time_yr']]),
            'saturation margin': lambda h: h['invariants'].update(min_saturation_margin_MPa=-0.1),
            'rate drift': lambda h: h['invariants'].update(rate_drift_fraction=0.05),
            'fluid system': lambda h: h['invariants'].update(fluid_system='CO2'),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                history = self._fixture()
                mutate(history)
                with self.assertRaises(ValueError):
                    self._reservoir_model(history)

    def test_horizon_must_cover_plant_or_integrity_lifetime(self):
        history = self._fixture()
        cut = 20 * _TIMESTEPS_PER_YEAR + 1
        for name in ('time_yr', 'T_feedzone_C', 'P_feedzone_MPa', 'h_feedzone_MJkg', 'mass_rate_kgs'):
            history[name] = history[name][:cut]
        history['scenario']['horizon_years'] = 20.0

        with self.assertRaises(ValueError) as cm:
            self._reservoir_model(history)
        self.assertIn('must cover 25 yr', str(cm.exception))

        # A well integrity lifetime shorter than the profile makes the shorter horizon sufficient
        model = self._reservoir_model(history, {'Well Integrity Maximum Lifetime': 10})
        self.assertEqual(len(model.reserv.Tresoutput.value), len(model.reserv.timevector.value))
        # ...and the last profile value is held over the remaining plant lifetime
        self.assertAlmostEqual(model.reserv.Tresoutput.value[-1], history['T_feedzone_C'][-1], delta=1e-9)

    # ------------------------------------------------------------------------------------------------------------
    # Redrilling with the reference window, tiling and the redrill cap (WellBores.calculate_redrilling hooks)
    # ------------------------------------------------------------------------------------------------------------

    def _redrill(self, history: dict, params: dict | None = None, produced_temperature_C: float | None = None) -> Model:
        """
        Run the redrilling calculation on the profile's temperature history directly (no wellbore solve). The produced
        temperature is the profile's, or the constant given.
        """
        model = self._reservoir_model(history, params)
        wellbores = model.wellbores
        wellbores.ProdTempDrop.value = 0.0
        if produced_temperature_C is None:
            wellbores.ProducedTemperature.value = np.array(model.reserv.Tresoutput.value, dtype=float)
        else:
            wellbores.ProducedTemperature.value = np.full(len(model.reserv.Tresoutput.value), produced_temperature_C)
        wellbores.calculate_redrilling(model)
        return model

    _FLAT_HISTORY_DEPTH_KM = 5.1225  # 200.0 C bottom-hole temperature at 36.7 C/km with a 12 C surface

    def _flat_history(self, T_C: float = 200.0) -> dict:
        history = self._fixture()
        history['scenario']['target_temperature_C'] = T_C
        history['scenario']['reservoir_depth_km'] = self._FLAT_HISTORY_DEPTH_KM
        history['T_feedzone_C'] = [T_C] * len(history['time_yr'])
        return history

    def test_window_max_reference_ignores_early_dip(self):
        """A 1.5 C dip in month 2 (0.75 % at 200 C, above Maximum Drawdown) neither triggers nor sets the reference."""
        history = self._flat_history()
        history['T_feedzone_C'][2] = 198.5
        model = self._redrill(history, {'Reservoir Depth': self._FLAT_HISTORY_DEPTH_KM})
        self.assertEqual(model.wellbores.redrill.value, 0)
        self.assertEqual(model.reserv.drawdown_reference_steps, 12)

    def test_genuine_decline_triggers_and_tiles_histories(self):
        history = self._flat_history()
        n = len(history['time_yr'])
        decline_from = 3 * _TIMESTEPS_PER_YEAR  # a 0.3 % decline from year 3 on
        history['T_feedzone_C'] = [200.0 if i < decline_from else 199.4 for i in range(n)]
        history['P_feedzone_MPa'] = [40.0 - 0.01 * i for i in range(n)]
        model = self._redrill(history, {'Reservoir Depth': self._FLAT_HISTORY_DEPTH_KM})
        wellbores = model.wellbores
        reserv = model.reserv

        total = len(reserv.timevector.value)
        resampled_T = np.interp(reserv.timevector.value, history['time_yr'], history['T_feedzone_C'])
        trigger = int(np.argmax(resampled_T < (1 - wellbores.maxdrawdown.value) * np.max(resampled_T[:12])))
        self.assertGreaterEqual(trigger, decline_from - 1)  # the GEOPHIRES grid (25/299 yr) is not the profile's
        self.assertLessEqual(trigger, decline_from + 1)
        self.assertEqual(wellbores.redrill.value, int(np.floor(total / trigger)))
        expected_T = np.tile(resampled_T[:trigger], wellbores.redrill.value + 1)[:total]
        np.testing.assert_allclose(reserv.Tresoutput.value, expected_T)
        np.testing.assert_allclose(wellbores.ProducedTemperature.value, expected_T)

        # the Fimbul diagnostics are tiled identically with the temperature
        resampled_P_kPa = np.interp(reserv.timevector.value, history['time_yr'], history['P_feedzone_MPa']) * 1000.0
        expected_P = np.tile(resampled_P_kPa[:trigger], wellbores.redrill.value + 1)[:total]
        np.testing.assert_allclose(reserv.fimbul_feedzone_pressure_kPa, expected_P)
        self.assertEqual(len(reserv.fimbul_feedzone_enthalpy_kJ_kg), total)
        self.assertEqual(len(reserv.fimbul_mass_rate_kg_s), total)

    def test_max_redrill_events_guard(self):
        history = self._flat_history()
        n = len(history['time_yr'])
        history['T_feedzone_C'] = [200.0 if i < 14 else 199.0 for i in range(n)]  # trigger right after the window
        with self.assertRaises(RuntimeError) as cm:
            self._redrill(history, {'Reservoir Depth': self._FLAT_HISTORY_DEPTH_KM, 'Fimbul Maximum Redrill Events': 3})
        self.assertIn('redrill events', str(cm.exception))
        self.assertIn('FimbulReservoir', str(cm.exception))

    def test_reference_window_of_zero_reproduces_the_standard_trigger(self):
        """Without the window the dip fires the standard trigger at its own time step; the redrill cap catches it."""
        history = self._flat_history()
        history['T_feedzone_C'][2] = 198.5
        with self.assertRaises(RuntimeError) as cm:
            self._redrill(
                history, {'Reservoir Depth': self._FLAT_HISTORY_DEPTH_KM, 'Fimbul Drawdown Reference Window': 0}
            )
        self.assertIn('time step 2 of 300', str(cm.exception))
        self.assertIn('more than the 12 allowed', str(cm.exception))

    def test_window_and_cap_act_on_the_trigger_temperature_basis(self):
        """
        The reference window and the redrill cap act on the temperature Redrilling Trigger Temperature names: a dip in
        the reservoir output temperature that the produced temperature does not show fires the reservoir output basis
        (which the coupled wellbore model always uses) but not the produced temperature basis.
        """
        history = self._flat_history()
        history['T_feedzone_C'][2] = 198.5
        params = {'Reservoir Depth': self._FLAT_HISTORY_DEPTH_KM, 'Fimbul Drawdown Reference Window': 0}

        with self.assertRaises(RuntimeError) as cm:
            self._redrill(history, params, produced_temperature_C=200.0)
        self.assertIn('time step 2 of 300', str(cm.exception))

        model = self._redrill(
            history, {**params, 'Production Wellbore Model': 1, 'Redrilling Trigger Temperature': 0}, 200.0
        )
        self.assertNotIsInstance(model.wellbores, CoupledWellBores)
        self.assertIs(model.wellbores.redrilling_trigger_temperature.value, RedrillingTriggerTemperature.PRODUCED)
        self.assertEqual(model.wellbores.redrill.value, 0)

    # ------------------------------------------------------------------------------------------------------------
    # Pressure-enthalpy water (H2O) profiles
    # ------------------------------------------------------------------------------------------------------------

    _H2O_REPORT_STEPS_PER_YEAR = (12, 4)
    """Report steps per year in the first year and after it, as in fimbul-python's parallel-fracture unit"""

    _H2O_DECLINE_C = 3.0
    _H2O_GRINGARTEN_DECLINE_C = 2.0

    def _h2o_profile_path(self, history: dict) -> str:
        """Write the profile into a temporary directory removed after the test."""
        tmp = tempfile.TemporaryDirectory(prefix='fimbul-h2o-')
        self.addCleanup(tmp.cleanup)
        return self._write_profile(history, tmp.name)

    def _h2o_reservoir_model(self, history: dict) -> Model:
        return self._reservoir_model(params={'Fimbul Feedzone Profile File': self._h2o_profile_path(history)})

    def _h2o_history(self, max_vapor_saturation: float, max_vapor_saturation_feedzone: float = 0.0) -> dict:
        """
        A small synthetic FeedzoneHistory of the kind fimbul-python's parallel-fracture unit writes from Fimbul's
        pressure-enthalpy water model, for the example's scenario: monthly report steps in the first year and quarterly
        ones after it, invariants with fluid_system H2O and the maximum vapor saturation but no saturation margin, a
        Gringarten block on the profile's own time axis and provenance without a case hash. The feedzone temperature
        declines linearly by _H2O_DECLINE_C over the plant lifetime, so that it is told apart from the example's
        Gringarten-shaped fixture.
        """
        scenario = dict(self._fixture()['scenario'])
        horizon_yr = float(scenario['horizon_years'])
        first, after = self._H2O_REPORT_STEPS_PER_YEAR
        time_yr = np.concatenate(
            [np.arange(0, first + 1) / first, 1.0 + np.arange(1, round((horizon_yr - 1.0) * after) + 1) / after]
        )
        T0_C = float(scenario['target_temperature_C'])
        T_C = T0_C - self._H2O_DECLINE_C * time_yr / horizon_yr
        gringarten_C = T0_C - self._H2O_GRINGARTEN_DECLINE_C * time_yr / horizon_yr
        P_MPa = np.full(len(time_yr), self._fixture()['P_feedzone_MPa'][0])
        h_MJkg = [CP.PropsSI('H', 'T', T + 273.15, 'P', P * 1e6, 'Water') / 1e6 for T, P in zip(T_C, P_MPa)]
        flow_kgs = float(scenario['flow_rate_per_well_kg_s'])
        return {
            'schema_version': '1',
            'source': 'fimbul',
            'scenario': scenario,
            'time_yr': [float(t) for t in time_yr],
            'T_feedzone_C': [float(T) for T in T_C],
            'P_feedzone_MPa': [float(P) for P in P_MPa],
            'h_feedzone_MJkg': [float(h) for h in h_MJkg],
            'mass_rate_kgs': [flow_kgs] * len(time_yr),
            'gringarten': {'Tresoutput_C': [float(T) for T in gringarten_C], 'mean_C': float(np.mean(gringarten_C))},
            'invariants': {
                'fluid_system': FLUID_SYSTEM_H2O,
                'max_vapor_saturation': max_vapor_saturation,
                'max_vapor_saturation_feedzone': max_vapor_saturation_feedzone,
                'min_pressure_MPa': float(np.min(P_MPa)),
                'max_pressure_MPa': float(np.max(P_MPa)),
                'rate_drift_fraction': 0.0,
                'horizon_ok': True,
            },
            'provenance': {
                'fimbul_version': '0.4.0',
                'jutuldarcy_version': '0.4.0',
                'jutul_version': '0.4.0',
                'fluid_system': FLUID_SYSTEM_H2O,
                'case': 'fracture_unit',
                'unit': {'cells': 240, 'fracture_cells': 24, 'fractures_per_production_well': 249.9},
                'spec': {'reservoir_depth_m': 5120.0, 'initial_temperature_C': T0_C, 'horizon_years': horizon_yr},
                'note': 'Synthetic pressure-enthalpy water profile written by '
                'tests/geophires_x_tests/test_fimbul_reservoir.py; not a Fimbul run.',
            },
        }

    def test_h2o_single_phase_profile(self):
        history = self._h2o_history(max_vapor_saturation=0.0)
        model = self._h2o_reservoir_model(history)
        reserv = model.reserv
        times = np.array(reserv.timevector.value, dtype=float)

        # the profile's temperature history, resampled from its uneven report steps, is the reservoir output
        expected_T = np.interp(times, history['time_yr'], history['T_feedzone_C'])
        np.testing.assert_allclose(reserv.Tresoutput.value, expected_T)
        self.assertAlmostEqual(
            reserv.Tresoutput.value[0] - reserv.Tresoutput.value[-1], self._H2O_DECLINE_C, delta=0.02
        )
        self.assertGreater(
            np.max(np.abs(np.array(reserv.Tresoutput.value) - np.array(self._fixture()['gringarten']['Tresoutput_C']))),
            1.0,
        )

        self.assertEqual(reserv.fimbul_fluid_system, FLUID_SYSTEM_H2O)
        self.assertEqual(reserv.fimbul_fluid_system_output.value, FLUID_SYSTEM_H2O)
        self.assertEqual(reserv.fimbul_max_vapor_saturation_output.value, 0.0)
        self.assertEqual(reserv.fimbul_max_vapor_saturation_feedzone, 0.0)
        # no saturation margin for a profile in which two-phase water is modelled
        self.assertTrue(math.isnan(reserv.fimbul_min_saturation_margin.value))
        self.assertEqual(reserv.fimbul_case_hash_output.value, '')
        self.assertAlmostEqual(reserv.fimbul_achieved_flow_rate.value, 60.0, delta=1e-9)
        for name in reserv.tiled_histories:
            self.assertEqual(len(getattr(reserv, name)), len(times))

        # the Gringarten block is on the profile's own time axis and is resampled onto the GEOPHIRES grid
        expected_delta = float(
            np.mean(expected_T - np.interp(times, history['time_yr'], history['gringarten']['Tresoutput_C']))
        )
        self.assertAlmostEqual(reserv.fimbul_gringarten_mean_delta.value, expected_delta, places=9)
        self.assertLess(reserv.fimbul_gringarten_mean_delta.value, -0.4)

    def test_h2o_two_phase_profile(self):
        """Two-phase water is modelled in an H2O profile: a positive vapor saturation is reported, not an error."""
        history = self._h2o_history(max_vapor_saturation=0.12)
        # a saturation margin, if present, is not checked for an H2O profile
        history['invariants']['min_saturation_margin_MPa'] = -0.5
        model = self._h2o_reservoir_model(history)
        self.assertEqual(model.reserv.fimbul_max_vapor_saturation_output.value, 0.12)
        self.assertTrue(math.isnan(model.reserv.fimbul_min_saturation_margin.value))

        # vapor at the feedzone itself, where GEOPHIRES takes only the temperature, is worth a warning
        history = self._h2o_history(max_vapor_saturation=0.12, max_vapor_saturation_feedzone=0.05)
        with self.assertLogs(level='WARNING') as logs:
            model = self._h2o_reservoir_model(history)
        self.assertTrue(any('feedzone water was two-phase' in it for it in logs.output))
        self.assertEqual(model.reserv.fimbul_max_vapor_saturation_feedzone, 0.05)
        np.testing.assert_allclose(
            model.reserv.Tresoutput.value,
            np.interp(model.reserv.timevector.value, history['time_yr'], history['T_feedzone_C']),
        )

    def test_h2o_invalid_documents_raise(self):
        cases = {
            'vapor saturation above 1': lambda h: h['invariants'].update(max_vapor_saturation=1.5),
            'negative feedzone vapor saturation': lambda h: h['invariants'].update(max_vapor_saturation_feedzone=-0.1),
            'rate drift': lambda h: h['invariants'].update(rate_drift_fraction=0.05),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                history = self._h2o_history(max_vapor_saturation=0.0)
                mutate(history)
                with self.assertRaises(ValueError):
                    self._h2o_reservoir_model(history)

    def _conventional_result(self, history: dict) -> GeophiresXResult:
        """The example run with Ramey's production wellbore model and a subcritical ORC plant (no superhot-wellbore)."""
        example_path = _without_inputs(self._example_path(), without_prefixes=_COUPLED_WELLBORE_INPUT_PREFIXES)
        params = {
            'Power Plant Type': 1,
            'Fimbul Feedzone Profile File': self._h2o_profile_path(history),
        }
        return GeophiresXClient().get_geophires_result(
            GeophiresInputParameters(from_file_path=example_path, params=params)
        )

    def test_h2o_profile_outputs(self):
        """The fluid system and the maximum vapor saturation replace the saturation margin in the case report."""
        reservoir_parameters = self._conventional_result(self._h2o_history(max_vapor_saturation=0.12)).result[
            'RESERVOIR PARAMETERS'
        ]
        self.assertEqual(reservoir_parameters['Reservoir Model'], 'Fimbul Feedzone Profile Model')
        self.assertEqual(reservoir_parameters['Fimbul fluid system']['value'], FLUID_SYSTEM_H2O)
        self.assertEqual(reservoir_parameters['Fimbul maximum vapor saturation']['value'], 0.12)
        self.assertIsNone(reservoir_parameters['Fimbul minimum saturation margin'])
        # the profile has no case hash: the case report line is left empty, which reads back as no value
        self.assertIsNone(reservoir_parameters['Fimbul case hash'])
        self.assertAlmostEqual(reservoir_parameters['Fimbul achieved flow rate per well']['value'], 60.0, delta=1e-6)

        # ...whereas a single-phase liquid profile reports its saturation margin as before
        reservoir_parameters = self._conventional_result(self._fixture()).result['RESERVOIR PARAMETERS']
        self.assertIsNone(reservoir_parameters['Fimbul fluid system'])
        self.assertIsNone(reservoir_parameters['Fimbul maximum vapor saturation'])
        self.assertGreater(reservoir_parameters['Fimbul minimum saturation margin']['value'], 0)
        self.assertEqual(reservoir_parameters['Fimbul case hash']['value'], 'synthetic-gringarten-example_SHR-6')

    # ------------------------------------------------------------------------------------------------------------
    # End to end
    # ------------------------------------------------------------------------------------------------------------

    @unittest.skipUnless(SUPERHOT_WELLBORE_AVAILABLE, 'superhot-wellbore is not installed')
    def test_example_matches_gringarten_example(self):
        """example_FIMBUL-1 (the synthetic Gringarten-shaped profile) reproduces example_SHR-6: the swap is pure."""
        client = GeophiresXClient()
        fimbul = client.get_geophires_result(GeophiresInputParameters(from_file_path=self._example_path()))
        gringarten = client.get_geophires_result(
            GeophiresInputParameters(from_file_path=self._get_test_file_path(Path('..', _GRINGARTEN_EXAMPLE_FILE)))
        )

        self.assertEqual(fimbul.result['RESERVOIR PARAMETERS']['Reservoir Model'], 'Fimbul Feedzone Profile Model')
        self.assertEqual(
            fimbul.result['RESERVOIR PARAMETERS']['Fimbul case hash']['value'], 'synthetic-gringarten-example_SHR-6'
        )
        delta = fimbul.result['RESERVOIR PARAMETERS']['Fimbul minus Gringarten mean produced temperature']
        self.assertEqual(delta['unit'], 'degC')
        self.assertLess(abs(delta['value']), 0.05)
        self.assertGreater(fimbul.result['RESERVOIR PARAMETERS']['Fimbul minimum saturation margin']['value'], 0)
        self.assertAlmostEqual(
            fimbul.result['RESERVOIR PARAMETERS']['Fimbul achieved flow rate per well']['value'], 60.0, delta=1e-6
        )

        for category, field in (
            ('SURFACE EQUIPMENT SIMULATION RESULTS', 'Average Net Electricity Generation'),
            ('SUMMARY OF RESULTS', 'Electricity breakeven price'),
            ('RESERVOIR SIMULATION RESULTS', 'Average Production Temperature'),
            ('RESERVOIR SIMULATION RESULTS', 'Production Pump Depth'),
        ):
            with self.subTest(field=field):
                self.assertAlmostEqualWithinPercentage(
                    gringarten.result[category][field]['value'], fimbul.result[category][field]['value'], percent=1
                )
        self.assertEqual(
            fimbul.result['ENGINEERING PARAMETERS']['Number of times redrilling']['value'],
            gringarten.result['ENGINEERING PARAMETERS']['Number of times redrilling']['value'],
        )

    @unittest.skipUnless(SUPERHOT_WELLBORE_AVAILABLE, 'superhot-wellbore is not installed')
    def test_example_result(self):
        """The example input reproduces its checked-in result (see also test_geophires_examples)."""
        client = GeophiresXClient()
        result = client.get_geophires_result(GeophiresInputParameters(from_file_path=self._example_path()))
        expected = self._strip_metadata(GeophiresXResult(self._example_path().replace('.txt', '.out')))
        self.assertDictEqual(expected.result, self._strip_metadata(result).result)

    @unittest.skipUnless(SUPERHOT_WELLBORE_AVAILABLE, 'superhot-wellbore is not installed')
    def test_flow_rate_must_be_prescribed(self):
        """The coupled wellbore model cannot solve for the flow rate of a profile simulated at a given flow rate."""
        history = self._fixture()
        history['scenario']['flow_rate_per_well_kg_s'] = 50.0  # the default Production Flow Rate per Well
        history['mass_rate_kgs'] = [50.0] * len(history['time_yr'])
        model = self._model(
            {'Production Wellhead Pressure': '1500 kPa', 'Fimbul Feedzone Profile File': self._write_profile(history)},
            without=('Production Flow Rate per Well',),
        )
        with self.assertRaises(ValueError) as cm:
            model.Calculate()
        self.assertIn('requires Production Flow Rate per Well', str(cm.exception))


if __name__ == '__main__':
    if '--regenerate' in sys.argv:
        print(f'Wrote {regenerate_example_fixture()}')
    else:
        unittest.main()
