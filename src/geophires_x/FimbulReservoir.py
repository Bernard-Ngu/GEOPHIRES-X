"""
Fimbul feedzone profile reservoir model (Reservoir Model, 9).

Reads the far-field (feedzone) temperature history of a production well that was simulated beforehand with Fimbul.jl
(JutulDarcy) through the fimbul-python package and written as a FeedzoneHistory JSON document. GEOPHIRES itself never
imports fimbul or Julia: the JSON is the whole hand-off, so the reservoir model runs wherever GEOPHIRES runs.

The document (schema_version "1") carries the scenario it was run for (reservoir depth, bottom-hole temperature, flow
rate per production well, fracture height), the series time_yr / T_feedzone_C / P_feedzone_MPa / h_feedzone_MJkg /
mass_rate_kgs starting at t = 0, the run-time invariants that make it trustworthy (rate drift, horizon and the phase
invariants below) and, optionally, the Gringarten temperature history of the same deck for comparison. The scenario is
checked against the deck, the temperature series is resampled onto the GEOPHIRES time vector (holding the last value)
and becomes the reservoir output temperature; everything else GEOPHIRES derives from the deck exactly as for the
Gringarten model (fracture geometry, reservoir volume, hydrostatic pressure, drawdown from the productivity index and
injectivity index). The pressure, enthalpy and mass rate series are exposed as diagnostics only
(fimbul_feedzone_pressure_kPa, fimbul_feedzone_enthalpy_kJ_kg, fimbul_mass_rate_kg_s), tiled alongside the temperature
when redrilling repeats it.

Two kinds of profile are read, told apart by invariants.fluid_system:

* No fluid_system: a single-phase liquid water run (Fimbul's EGS case). Its validity requires the pressure to stay above
  the saturation pressure everywhere, so a non-positive invariants.min_saturation_margin_MPa is an error; the margin is
  reported as Fimbul Minimum Saturation Margin.
* fluid_system "H2O": a run on Fimbul's pressure-enthalpy water model (fimbul-python's parallel-fracture unit, Fimbul
  0.4), which carries liquid, two-phase, vapor and supercritical water through the steam tables. Two-phase water is
  modelled there, not an error, so there is no saturation margin; the largest vapor saturation over the run
  (invariants.max_vapor_saturation, 0 when the water stayed single-phase everywhere) is reported as Fimbul Maximum Vapor
  Saturation instead, and the fluid system as Fimbul Fluid System.

Redrilling: a numerically simulated profile can have an early warm-up hump or dip, so the thermal drawdown trigger of
WellBores.calculate_redrilling measures against the maximum trigger temperature (Redrilling Trigger Temperature) over
Fimbul Drawdown Reference Window (and cannot fire inside it), and a redrill count above Fimbul Maximum Redrill Events is
an error rather than a constant temperature tiled hundreds of times. The profile must cover the shorter of the plant
lifetime and Well Integrity Maximum Lifetime so that no redrilling trigger falls in a held extrapolation.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from .Parameter import floatParameter, intParameter, strParameter, OutputParameter
from .Units import Units, TimeUnit, TemperatureUnit, PressureUnit, FlowRateUnit
import geophires_x.Model as Model
from .Reservoir import Reservoir

FEEDZONE_HISTORY_SCHEMA_VERSION = '1'

FIMBUL_PROFILES_DIR_ENV_VAR = 'FIMBUL_PROFILES_DIR'

FLUID_SYSTEM_H2O = 'H2O'
"""invariants.fluid_system of a profile from Fimbul's pressure-enthalpy water model, in which two-phase water is
modelled; a profile without fluid_system is a single-phase liquid run"""

_SERIES_FIELDS = ('time_yr', 'T_feedzone_C', 'P_feedzone_MPa', 'h_feedzone_MJkg', 'mass_rate_kgs')

_VAPOR_SATURATION_INVARIANTS = ('max_vapor_saturation', 'max_vapor_saturation_feedzone')

_MAX_RATE_DRIFT_FRACTION = 0.02


class FimbulReservoir(Reservoir):
    """
    Reservoir model reading a Fimbul FeedzoneHistory JSON profile. See module docstring.
    """

    def __init__(self, model: Model):
        model.logger.info(f'Init {str(__class__)}: {sys._getframe().f_code.co_name}')
        super().__init__(model)
        sclass = str(__class__).replace("<class '", '')
        self.MyClass = sclass.replace("'>", '')

        self.provides_feedzone_history = True
        """Tells the coupled inflow-wellbore production wellbore model that the temperature history is externally
        simulated"""
        self.drawdown_reference_steps: int = 1
        """Read by WellBores.calculate_redrilling: reference window (time steps) of the thermal drawdown trigger"""
        self.tiled_histories: tuple = ()
        """Read by WellBores.calculate_redrilling: array attributes to tile alongside the temperature history"""
        self.max_redrill_events = None
        """Read by WellBores.calculate_redrilling: largest admissible redrill count"""

        self.fimbul_feedzone_pressure_kPa = np.empty(0)
        """Feedzone pressure history from Fimbul on the GEOPHIRES time vector [kPa] (diagnostic)"""
        self.fimbul_feedzone_enthalpy_kJ_kg = np.empty(0)
        """Feedzone specific enthalpy history from Fimbul on the GEOPHIRES time vector [kJ/kg] (diagnostic)"""
        self.fimbul_mass_rate_kg_s = np.empty(0)
        """Achieved mass rate per production well from Fimbul on the GEOPHIRES time vector [kg/s] (diagnostic)"""
        self.fimbul_case_hash: str = ''
        self.fimbul_fluid_system: str = ''
        """invariants.fluid_system of the profile: 'H2O' for the pressure-enthalpy water model, '' for a single-phase
        liquid profile"""
        self.fimbul_min_saturation_margin_kPa: float = float('nan')
        """Single-phase liquid profiles only"""
        self.fimbul_max_vapor_saturation: float = float('nan')
        """H2O profiles only: largest vapor saturation of any cell over the run"""
        self.fimbul_max_vapor_saturation_feedzone: float = float('nan')
        """H2O profiles only: largest vapor saturation of the producer's feedzone cell over the run"""
        self.fimbul_gringarten_mean_delta_C: float = float('nan')
        self.fimbul_profile_path: str = ''
        """Resolved path of the profile that was read, available after Calculate()"""

        self.profile_file = self.ParameterDict[self.profile_file.Name] = strParameter(
            'Fimbul Feedzone Profile File',
            DefaultValue='',
            UnitType=Units.NONE,
            ErrMessage='no Fimbul feedzone profile file given',
            ToolTipText='FeedzoneHistory JSON document written by fimbul-python (schema_version 1) with the feedzone '
            'temperature history of one production well simulated with Fimbul.jl for this deck: a single-phase '
            f'liquid profile, or a pressure-enthalpy water profile (invariants.fluid_system {FLUID_SYSTEM_H2O}). An '
            f'absolute path, or a path relative to the directory in the {FIMBUL_PROFILES_DIR_ENV_VAR} environment '
            'variable if set, else relative to the working directory.',
        )

        self.drawdown_reference_window = self.ParameterDict[self.drawdown_reference_window.Name] = floatParameter(
            'Fimbul Drawdown Reference Window',
            DefaultValue=1.0,
            Min=0.0,
            Max=100.0,
            UnitType=Units.TIME,
            PreferredUnits=TimeUnit.YEAR,
            CurrentUnits=TimeUnit.YEAR,
            ToolTipText='The thermal drawdown redrilling trigger (Maximum Drawdown) measures against the maximum '
            'temperature (per Redrilling Trigger Temperature) over this window from the start of production, and '
            'cannot fire inside it, so that the warm-up hump or dip of a numerically simulated profile neither sets '
            'the reference nor triggers redrilling.',
        )

        self.max_redrill_events_param = self.ParameterDict[self.max_redrill_events_param.Name] = intParameter(
            'Fimbul Maximum Redrill Events',
            DefaultValue=12,
            AllowableRange=list(range(0, 1001)),
            UnitType=Units.NONE,
            ToolTipText='Largest number of redrilling events admissible with a Fimbul profile; a thermal drawdown '
            'or well integrity trigger implying more is an error, since it would tile a short profile many times.',
        )

        self.initial_temperature_tolerance = self.ParameterDict[self.initial_temperature_tolerance.Name] = (
            floatParameter(
                'Fimbul Initial Temperature Tolerance',
                DefaultValue=5.0,
                Min=0.0,
                Max=100.0,
                UnitType=Units.TEMPERATURE,
                PreferredUnits=TemperatureUnit.CELSIUS,
                CurrentUnits=TemperatureUnit.CELSIUS,
                ToolTipText='A difference between the profile temperature at t = 0 and the bottom-hole temperature of '
                'the deck above this tolerance logs a warning; above four times it the run aborts.',
            )
        )

        self.fimbul_case_hash_output = self.OutputParameterDict[self.fimbul_case_hash_output.Name] = OutputParameter(
            Name='Fimbul Case Hash',
            display_name='Fimbul case hash',
            value='',
            UnitType=Units.NONE,
            ToolTipText='Case hash of the Fimbul run that produced the feedzone profile (provenance)',
        )

        self.fimbul_fluid_system_output = self.OutputParameterDict[self.fimbul_fluid_system_output.Name] = (
            OutputParameter(
                Name='Fimbul Fluid System',
                display_name='Fimbul fluid system',
                value='',
                UnitType=Units.NONE,
                ToolTipText=f'Fluid system of the Fimbul run (invariants.fluid_system): {FLUID_SYSTEM_H2O} for '
                'Fimbul\'s pressure-enthalpy water model, in which two-phase water is modelled; empty for a '
                'single-phase liquid profile',
            )
        )

        self.fimbul_min_saturation_margin = self.OutputParameterDict[self.fimbul_min_saturation_margin.Name] = (
            OutputParameter(
                Name='Fimbul Minimum Saturation Margin',
                display_name='Fimbul minimum saturation margin',
                UnitType=Units.PRESSURE,
                PreferredUnits=PressureUnit.KPASCAL,
                CurrentUnits=PressureUnit.KPASCAL,
                ToolTipText='Smallest margin of the pressure above the saturation pressure over all reservoir and '
                'fracture cells and report steps of the Fimbul run; the single-phase reservoir model is valid only '
                f'while it is positive. Single-phase liquid profiles only (NaN for {FLUID_SYSTEM_H2O} profiles)',
            )
        )

        self.fimbul_max_vapor_saturation_output = self.OutputParameterDict[
            self.fimbul_max_vapor_saturation_output.Name
        ] = OutputParameter(
            Name='Fimbul Maximum Vapor Saturation',
            display_name='Fimbul maximum vapor saturation',
            UnitType=Units.NONE,
            ToolTipText='Largest vapor saturation (fraction of the pore volume filled with steam, 0 to 1) over all '
            'reservoir and fracture cells and report steps of the Fimbul run; 0 when the water stayed single-phase '
            f'everywhere. {FLUID_SYSTEM_H2O} (pressure-enthalpy water) profiles only (NaN for single-phase liquid '
            'profiles)',
        )

        self.fimbul_achieved_flow_rate = self.OutputParameterDict[self.fimbul_achieved_flow_rate.Name] = (
            OutputParameter(
                Name='Fimbul Achieved Flow Rate per Well',
                display_name='Fimbul achieved flow rate per well',
                UnitType=Units.FLOWRATE,
                PreferredUnits=FlowRateUnit.KGPERSEC,
                CurrentUnits=FlowRateUnit.KGPERSEC,
                ToolTipText='Mean mass rate per production well achieved by the rate-controlled producer of the '
                'Fimbul run',
            )
        )

        self.fimbul_gringarten_mean_delta = self.OutputParameterDict[self.fimbul_gringarten_mean_delta.Name] = (
            OutputParameter(
                Name='Fimbul Minus Gringarten Mean Produced Temperature',
                display_name='Fimbul minus Gringarten mean produced temperature',
                UnitType=Units.TEMPERATURE,
                PreferredUnits=TemperatureUnit.CELSIUS,
                CurrentUnits=TemperatureUnit.CELSIUS,
                ToolTipText='Mean over the plant lifetime of the Fimbul feedzone temperature minus the Gringarten '
                'reservoir output temperature of the same deck (the shadow curve stored in the profile), before '
                'redrilling; NaN when the profile carries no Gringarten block',
            )
        )

        model.logger.info(f'Complete {str(__class__)}: {sys._getframe().f_code.co_name}')

    def __str__(self):
        return 'FimbulReservoir'

    def read_parameters(self, model: Model) -> None:
        model.logger.info(f'Init {str(__class__)}: {sys._getframe().f_code.co_name}')
        super().read_parameters(model)

        if not str(self.profile_file.value).strip():
            raise ValueError(
                f'{self.resoption.Name} {self.resoption.value.int_value} ({self.resoption.value.value}) requires '
                f'{self.profile_file.Name}.'
            )

        model.logger.info(f'Complete {str(__class__)}: {sys._getframe().f_code.co_name}')

    def Calculate(self, model: Model) -> None:
        """
        Run the base reservoir calculation (geometry, bottom-hole temperature, time vector, fluid properties), then
        read the profile, check it against the deck and set the reservoir output temperature from it. The whole
        temperature history is overwritten on every call because the base calculation is cached and redrilling
        modifies the history in place.
        """
        model.logger.info(f'Init {str(__class__)}: {sys._getframe().f_code.co_name}')
        super().Calculate(model)

        path = self.resolve_profile_path(str(self.profile_file.value))
        self.fimbul_profile_path = str(path)
        with open(path, encoding='utf-8') as f:
            history = json.load(f)
        self._validate_history(history, path)
        self._check_scenario(model, history, path)

        times = np.array(self.timevector.value, dtype=float)
        time_yr = np.array(history['time_yr'], dtype=float)

        def _resample(values) -> np.ndarray:
            return np.interp(times, time_yr, np.array(values, dtype=float))

        # Only the temperature enters the calculation; the hydraulics stay on the productivity and injectivity index.
        self.Tresoutput.value = _resample(history['T_feedzone_C'])
        self.fimbul_feedzone_pressure_kPa = _resample(history['P_feedzone_MPa']) * 1000.0
        self.fimbul_feedzone_enthalpy_kJ_kg = _resample(history['h_feedzone_MJkg']) * 1000.0
        self.fimbul_mass_rate_kg_s = _resample(history['mass_rate_kgs'])

        timesteps_per_year = int(model.economics.timestepsperyear.value)
        window_steps = int(round(float(self.drawdown_reference_window.value) * timesteps_per_year))
        self.drawdown_reference_steps = max(window_steps, 1)
        self.tiled_histories = (
            'fimbul_feedzone_pressure_kPa',
            'fimbul_feedzone_enthalpy_kJ_kg',
            'fimbul_mass_rate_kg_s',
        )
        self.max_redrill_events = int(self.max_redrill_events_param.value)

        provenance = history.get('provenance') or {}
        invariants = history.get('invariants') or {}
        self.fimbul_case_hash = str(provenance.get('case_hash') or '')
        self.fimbul_fluid_system = str(invariants.get('fluid_system') or '')
        if self.fimbul_fluid_system == FLUID_SYSTEM_H2O:
            self.fimbul_min_saturation_margin_kPa = float('nan')
            self.fimbul_max_vapor_saturation = self._optional_float(invariants.get('max_vapor_saturation'))
            self.fimbul_max_vapor_saturation_feedzone = self._optional_float(
                invariants.get('max_vapor_saturation_feedzone')
            )
            phase_summary = self._log_vapor_saturation(model, path)
        else:
            margin_MPa = invariants.get('min_saturation_margin_MPa')
            self.fimbul_min_saturation_margin_kPa = (
                float(margin_MPa) * 1000.0 if margin_MPa is not None else float('nan')
            )
            self.fimbul_max_vapor_saturation = float('nan')
            self.fimbul_max_vapor_saturation_feedzone = float('nan')
            phase_summary = (
                'single-phase liquid profile, minimum saturation margin '
                f'{self.fimbul_min_saturation_margin_kPa:.0f} kPa'
            )
        self.fimbul_gringarten_mean_delta_C = self._gringarten_mean_delta_C(history)

        self.fimbul_case_hash_output.value = self.fimbul_case_hash
        self.fimbul_fluid_system_output.value = self.fimbul_fluid_system
        self.fimbul_min_saturation_margin.value = self.fimbul_min_saturation_margin_kPa
        self.fimbul_max_vapor_saturation_output.value = self.fimbul_max_vapor_saturation
        self.fimbul_achieved_flow_rate.value = float(np.mean(self.fimbul_mass_rate_kg_s))
        self.fimbul_gringarten_mean_delta.value = self.fimbul_gringarten_mean_delta_C

        versions = ', '.join(
            f'{key} {provenance[key]}'
            for key in ('fimbul_version', 'jutuldarcy_version', 'jutul_version')
            if provenance.get(key)
        )
        model.logger.info(
            f'Fimbul feedzone profile {path} ({phase_summary}'
            + (f'; case {provenance["case"]}' if provenance.get('case') else '')
            + (f'; {versions}' if versions else '')
            + f'): {len(time_yr)} samples over {time_yr[-1]:.2f} yr resampled onto {len(times)} time steps; feedzone '
            f'temperature {self.Tresoutput.value[0]:.1f} to {self.Tresoutput.value[-1]:.1f} C, Fimbul minus '
            f'Gringarten mean {self.fimbul_gringarten_mean_delta_C:.2f} C'
        )
        model.logger.info(f'Complete {str(__class__)}: {sys._getframe().f_code.co_name}')

    def _log_vapor_saturation(self, model: Model, path: Path) -> str:
        """
        Log what the vapor saturation of an H2O profile means for GEOPHIRES and return a summary for the profile log
        line. GEOPHIRES takes only the feedzone temperature from the profile, so vapor at the feedzone is worth a
        warning: there the temperature is the saturation temperature and does not by itself define the produced state.
        """
        if not math.isfinite(self.fimbul_max_vapor_saturation):
            msg = (
                f'Fimbul feedzone profile {path}: fluid system {self.fimbul_fluid_system} profile has no '
                'invariants.max_vapor_saturation; Fimbul Maximum Vapor Saturation is not reported.'
            )
            print(f'Warning: {msg}')
            model.logger.warning(msg)
        elif self.fimbul_max_vapor_saturation_feedzone > 0:
            msg = (
                f'Fimbul feedzone profile {path}: the feedzone water was two-phase during the Fimbul run (vapor '
                f'saturation up to {self.fimbul_max_vapor_saturation_feedzone:.4f} at the feedzone, '
                f'{self.fimbul_max_vapor_saturation:.4f} anywhere). GEOPHIRES uses only the feedzone temperature; '
                'the productivity-index hydraulics and the production wellbore model assume single-phase inflow.'
            )
            print(f'Warning: {msg}')
            model.logger.warning(msg)

        feedzone = (
            f' ({self.fimbul_max_vapor_saturation_feedzone:.4f} at the feedzone)'
            if math.isfinite(self.fimbul_max_vapor_saturation_feedzone)
            else ''
        )
        return (
            f'fluid system {self.fimbul_fluid_system}, maximum vapor saturation '
            f'{self.fimbul_max_vapor_saturation:.4f}{feedzone}'
        )

    @staticmethod
    def _optional_float(value) -> float:
        return float(value) if value is not None else float('nan')

    # ------------------------------------------------------------------------------------------------------------
    # Profile reading and validation
    # ------------------------------------------------------------------------------------------------------------

    @staticmethod
    def resolve_profile_path(profile_file: str) -> Path:
        """
        Absolute path of the profile: the path as given if absolute, else relative to FIMBUL_PROFILES_DIR when set,
        else relative to the working directory (and, as a convenience for the bundled examples, to the GEOPHIRES
        package and source tree).
        """
        given = Path(profile_file).expanduser()
        if given.is_absolute():
            candidates = [given]
        else:
            candidates = []
            profiles_dir = os.environ.get(FIMBUL_PROFILES_DIR_ENV_VAR)
            if profiles_dir:
                candidates.append(Path(profiles_dir, given))
            package_dir = Path(__file__).resolve().parent
            candidates.extend([Path.cwd() / given, package_dir / given, package_dir.parent.parent / given])
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f'Fimbul Feedzone Profile File {profile_file!r} not found; looked in: '
            + ', '.join(str(c) for c in candidates)
            + f'. Give an absolute path or set {FIMBUL_PROFILES_DIR_ENV_VAR}.'
        )

    @staticmethod
    def _validate_history(history: dict, path: Path) -> None:
        if not isinstance(history, dict):
            raise ValueError(f'Fimbul feedzone profile {path} is not a JSON object')
        version = str(history.get('schema_version', ''))
        if version != FEEDZONE_HISTORY_SCHEMA_VERSION:
            raise ValueError(
                f'Fimbul feedzone profile {path} has schema_version {version!r}; '
                f'{FEEDZONE_HISTORY_SCHEMA_VERSION!r} is supported'
            )
        missing = [name for name in _SERIES_FIELDS if name not in history]
        if missing:
            raise ValueError(f'Fimbul feedzone profile {path} lacks {", ".join(missing)}')

        n = len(history['time_yr'])
        if n < 2:
            raise ValueError(f'Fimbul feedzone profile {path} has {n} samples; at least t = 0 and one report step')
        for name in _SERIES_FIELDS:
            values = np.array(history[name], dtype=float)
            if len(values) != n:
                raise ValueError(f'Fimbul feedzone profile {path}: {name} has {len(values)} samples, time_yr has {n}')
            if not np.all(np.isfinite(values)):
                raise ValueError(f'Fimbul feedzone profile {path}: {name} has non-finite values')
        time_yr = np.array(history['time_yr'], dtype=float)
        if abs(time_yr[0]) > 1e-12:
            raise ValueError(f'Fimbul feedzone profile {path}: time_yr must start at 0 (got {time_yr[0]})')
        if not np.all(np.diff(time_yr) > 0):
            raise ValueError(f'Fimbul feedzone profile {path}: time_yr is not strictly increasing')

        invariants = history.get('invariants') or {}
        fluid_system = invariants.get('fluid_system')
        if fluid_system is None:
            # Single-phase liquid run: valid only while every cell stays above the saturation pressure.
            margin = invariants.get('min_saturation_margin_MPa')
            if margin is not None and not margin > 0:
                raise ValueError(
                    f'Fimbul feedzone profile {path}: minimum saturation margin {margin} MPa is not positive, a '
                    'reservoir or fracture cell fell below saturation during the run'
                )
        elif fluid_system == FLUID_SYSTEM_H2O:
            # Pressure-enthalpy water: two-phase water is modelled, so there is no saturation margin to check.
            for key in _VAPOR_SATURATION_INVARIANTS:
                value = invariants.get(key)
                if value is not None and not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f'Fimbul feedzone profile {path}: {key} {value} is not a saturation between 0 and 1'
                    )
        else:
            raise ValueError(
                f'Fimbul feedzone profile {path} has fluid_system {fluid_system!r}; {FLUID_SYSTEM_H2O!r} '
                '(pressure-enthalpy water) and profiles without fluid_system (single-phase liquid) are supported'
            )
        drift = invariants.get('rate_drift_fraction')
        if drift is not None and drift > _MAX_RATE_DRIFT_FRACTION:
            raise ValueError(
                f'Fimbul feedzone profile {path}: the achieved rate drifted by {drift:.1%} over the run '
                f'(limit {_MAX_RATE_DRIFT_FRACTION:.0%}); it is not a constant-rate history'
            )

    def _check_scenario(self, model: Model, history: dict, path: Path) -> None:
        """
        The profile was simulated for a specific reservoir depth, bottom-hole temperature, flow rate per production
        well and fracture height: they must agree with the deck, since GEOPHIRES cannot re-run Fimbul.
        """
        scenario = history.get('scenario') or {}
        depth_m = float(self.depth.quantity().to('m').magnitude)
        checks = [
            ('reservoir_depth_km', lambda v: float(v) * 1000.0, depth_m, 1.0, 'm', f'{self.depth.Name}'),
            ('target_temperature_C', float, float(self.Trock.value), 0.5, 'degC', 'bottom-hole temperature'),
            (
                'flow_rate_per_well_kg_s',
                float,
                float(model.wellbores.prodwellflowrate.value),
                1e-6 * max(abs(float(model.wellbores.prodwellflowrate.value)), 1.0),
                'kg/s',
                model.wellbores.prodwellflowrate.Name,
            ),
            ('fracture_height_m', float, float(self.fracheightcalc.value), 0.01, 'm', 'fracture height'),
        ]
        for key, convert, deck_value, tolerance, unit, label in checks:
            if key not in scenario or scenario[key] is None:
                model.logger.warning(f'Fimbul feedzone profile {path}: scenario has no {key}; {label} not checked')
                continue
            profile_value = convert(scenario[key])
            if abs(profile_value - deck_value) > tolerance:
                raise ValueError(
                    f'Fimbul feedzone profile {path} was simulated for {label} {profile_value:g} {unit} but the '
                    f'deck has {deck_value:g} {unit}; the profile does not belong to this scenario.'
                )

        T0_profile = float(history['T_feedzone_C'][0])
        tolerance = float(self.initial_temperature_tolerance.value)
        delta = abs(T0_profile - float(self.Trock.value))
        if delta > 4.0 * tolerance:
            raise ValueError(
                f'Fimbul feedzone profile {path}: temperature at t = 0 ({T0_profile:.1f} C) differs from the '
                f'bottom-hole temperature ({float(self.Trock.value):.1f} C) by more than four times '
                f'{self.initial_temperature_tolerance.Name} ({tolerance:g} C).'
            )
        if delta > tolerance:
            msg = (
                f'Fimbul feedzone profile {path}: temperature at t = 0 ({T0_profile:.1f} C) differs from the '
                f'bottom-hole temperature ({float(self.Trock.value):.1f} C) by {delta:.1f} C.'
            )
            print(f'Warning: {msg}')
            model.logger.warning(msg)

        horizon_required_yr = float(model.surfaceplant.plant_lifetime.value)
        integrity = model.wellbores.well_integrity_max_lifetime
        if integrity.Provided and integrity.value > 0:
            horizon_required_yr = min(horizon_required_yr, float(integrity.quantity().to('yr').magnitude))
        horizon_yr = float(history['time_yr'][-1])
        if horizon_yr < horizon_required_yr - 1e-6:
            raise ValueError(
                f'Fimbul feedzone profile {path} ends at {horizon_yr:g} yr but must cover {horizon_required_yr:g} yr '
                f'(the shorter of {model.surfaceplant.plant_lifetime.Name} and {integrity.Name}) so that no '
                'redrilling trigger falls in a held extrapolation.'
            )

    def _gringarten_mean_delta_C(self, history: dict) -> float:
        gringarten = history.get('gringarten') or {}
        shadow = gringarten.get('Tresoutput_C') if isinstance(gringarten, dict) else None
        if not shadow:
            return float('nan')
        shadow = np.array(shadow, dtype=float)
        fimbul = np.array(self.Tresoutput.value, dtype=float)
        if self.fimbul_fluid_system == FLUID_SYSTEM_H2O and len(shadow) == len(history['time_yr']):
            # fimbul-python's parallel-fracture unit stores the shadow curve on the profile's own time axis.
            shadow = np.interp(np.array(self.timevector.value, dtype=float), history['time_yr'], shadow)
        elif len(shadow) != len(fimbul):
            # The shadow curve is on the GEOPHIRES grid of the deck that produced it; compare on the common length.
            n = min(len(shadow), len(fimbul))
            shadow, fimbul = shadow[:n], fimbul[:n]
        delta = float(np.mean(fimbul - shadow))
        return delta if math.isfinite(delta) else float('nan')
