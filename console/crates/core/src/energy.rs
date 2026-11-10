//! Measured power telemetry and a persisted daily energy ledger.
//!
//! This module only reads exposed OS interfaces and a local JSON file. It never
//! writes hardware, guesses a missing sensor or turns an unavailable reading
//! into zero. There is no whole-machine power sensor on every laptop: battery
//! discharge power is the only reading that covers the complete machine, while
//! AC input power and adapter losses may be unavailable. The ledger therefore
//! keeps counter-based CPU/package energy, integrated component energy and
//! battery discharge energy separate instead of inventing a total.
use crate::{CapabilityStatus, ControlError, EvidenceSource, Identity};
use serde::{Deserialize, Serialize};
#[cfg(any(target_os = "linux", test))]
use std::collections::HashSet;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

pub const LEDGER_SCHEMA_VERSION: u32 = 1;
pub const MAX_LEDGER_BYTES: u64 = 64 * 1024;
pub const HISTORY_DAYS: usize = 32;
/// Instantaneous components are only integrated across short intervals. Longer
/// gaps are still covered exactly by RAPL/battery energy counters.
pub const MAX_COMPONENT_GAP_SECONDS: f64 = 60.0;
pub const SHARED_LEDGER_PATH: &str = "/var/lib/mechrevo-console/energy.json";
/// A sampler is considered live while its last observation is this recent.
pub const LEDGER_FRESH_SECONDS: u64 = 120;
#[cfg(any(target_os = "linux", test))]
const MAX_POWER_WATTS: f64 = 10_000.0;
const MAX_DAY_WH: f64 = 100_000.0;
const MICROJOULES_PER_WH: f64 = 3_600_000_000.0;

/// One exposed power source. `energy_uj` counters are cumulative; `watts` is an
/// instantaneous reading that must not be multiplied across an arbitrary gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerSource {
    pub id: String,
    pub label: String,
    /// `rapl_package`, `rapl_platform`, `hwmon_power` or `battery_discharge`.
    pub kind: String,
    pub watts: Option<f64>,
    /// RAPL cumulative energy counter in microjoules.
    pub energy_uj: Option<u64>,
    pub max_energy_uj: Option<u64>,
    /// Battery remaining energy in watt-hours, derived from energy_now or
    /// charge_now * voltage_now. This is a state, not a consumption counter.
    pub energy_wh: Option<f64>,
    pub device_id: String,
    pub source: EvidenceSource,
    pub status: CapabilityStatus,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyStatus {
    pub sources: Vec<PowerSource>,
    /// True when any mains/USB supply reports online, false when all report
    /// offline, null when no external supply is exposed or readable.
    pub external_power: Option<bool>,
    pub battery_status: Option<String>,
    /// Sum of readable hwmon component power. Excludes battery and RAPL.
    pub components_watts: Option<f64>,
    /// Whole-machine battery discharge power, only while discharging.
    pub battery_watts: Option<f64>,
    pub notes: Vec<String>,
}

impl EnergyStatus {
    /// The best current value for a headline. Battery discharge covers the
    /// whole machine; component power is only a partial AC measurement.
    pub fn headline_watts(&self) -> Option<f64> {
        self.battery_watts.or(self.components_watts)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyDay {
    /// RAPL package/platform energy. Subdomains are never added to this value.
    pub cpu_wh: f64,
    /// Integrated hwmon component power, skipped while RAPL platform energy or
    /// battery discharge already covers the interval.
    pub components_wh: f64,
    /// Whole-machine battery discharge energy.
    pub battery_wh: f64,
    pub samples: u64,
    /// Seconds with a valid short-interval component integration.
    pub measured_seconds: u64,
    /// Coverage is tracked separately from the amount. A zero without these
    /// flags means "not available", not "measured zero".
    #[serde(default)]
    pub cpu_measured: bool,
    #[serde(default)]
    pub components_measured: bool,
    #[serde(default)]
    pub battery_measured: bool,
}

impl EnergyDay {
    pub fn total_wh(&self) -> f64 {
        self.cpu_wh + self.components_wh + self.battery_wh
    }

    /// The first observation only establishes counter/interval baselines. A
    /// zero total before that is not a measured zero-consumption day.
    pub fn has_measurement(&self) -> bool {
        self.cpu_measured || self.components_measured || self.battery_measured
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CounterState {
    pub energy_uj: u64,
    pub max_uj: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LastWatts {
    pub cpu_watts: Option<f64>,
    pub component_watts: Option<f64>,
    pub battery_watts: Option<f64>,
    pub at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenState {
    pub date: String,
    pub last_wall_ms: Option<u64>,
    pub ac_online: Option<bool>,
    pub battery_status: Option<String>,
    pub rapl: BTreeMap<String, CounterState>,
    pub battery_wh: Option<f64>,
    pub components_watts: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyLedger {
    pub schema_version: u32,
    /// DMI identity fingerprint. A different machine resets this ledger.
    pub machine: String,
    pub days: BTreeMap<String, EnergyDay>,
    #[serde(default)]
    pub open: OpenState,
    #[serde(default)]
    pub last: LastWatts,
}

impl EnergyLedger {
    pub fn new(machine: &str) -> Self {
        Self {
            schema_version: LEDGER_SCHEMA_VERSION,
            machine: machine.into(),
            days: BTreeMap::new(),
            open: OpenState::default(),
            last: LastWatts::default(),
        }
    }

    pub fn today(&self, date: &str) -> Option<&EnergyDay> {
        self.days.get(date)
    }

    /// Record one snapshot. Counter deltas are exact across long gaps; only
    /// instantaneous component power is limited to `MAX_COMPONENT_GAP_SECONDS`.
    pub fn observe(&mut self, status: &EnergyStatus, machine: &str, now: SystemTime) {
        let Some(now_ms) = unix_millis(now) else {
            return;
        };
        self.observe_on(status, machine, &local_date(now), now_ms);
    }

    /// Testable form with an explicit local date and wall-clock milliseconds.
    pub fn observe_on(&mut self, status: &EnergyStatus, machine: &str, date: &str, now_ms: u64) {
        if machine.is_empty() || !valid_date(date) {
            return;
        }
        if self.machine != machine {
            *self = Self::new(machine);
        }
        if !self.open.date.is_empty() && self.open.date != date {
            self.open = OpenState {
                date: date.into(),
                ..OpenState::default()
            };
            self.last = LastWatts::default();
        }
        let elapsed = self
            .open
            .last_wall_ms
            .and_then(|last| now_ms.checked_sub(last))
            .filter(|ms| *ms > 0)
            .map(|ms| ms as f64 / 1000.0);
        let short_interval = elapsed.is_some_and(|seconds| seconds <= MAX_COMPONENT_GAP_SECONDS);
        let battery = status
            .sources
            .iter()
            .find(|source| {
                source.kind == "battery_discharge" && source.status == CapabilityStatus::Available
            })
            .cloned();
        let discharging = battery.is_some()
            || status
                .battery_status
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("Discharging"))
            || status.external_power == Some(false);
        let day = self.days.entry(date.to_owned()).or_default();
        day.samples = day.samples.saturating_add(1);
        let mut battery_measured = false;

        if discharging {
            if let (Some(previous), Some(current)) = (
                self.open.battery_wh,
                battery.as_ref().and_then(|source| source.energy_wh),
            ) {
                if current < previous {
                    let delta = previous - current;
                    // A battery state difference is exact energy, but reject a
                    // corrupt jump rather than crediting an absurd day.
                    if delta.is_finite() && delta > 0.0 && delta <= MAX_DAY_WH {
                        day.battery_wh = add_wh(day.battery_wh, delta);
                        day.battery_measured = true;
                        battery_measured = true;
                    }
                }
            }
            if !battery_measured && short_interval {
                if let (Some(watts), Some(seconds)) = (status.battery_watts, elapsed) {
                    day.battery_wh = add_wh(day.battery_wh, watts * seconds / 3600.0);
                    day.battery_measured = true;
                }
            }
        }

        let platform_counter = status.sources.iter().any(|source| {
            source.kind == "rapl_platform"
                && source.energy_uj.is_some()
                && source.max_energy_uj.is_some()
        });
        let mut rapl_delta_uj = 0_u64;
        let mut rapl_observed = false;
        let mut next_counters = BTreeMap::new();
        for source in &status.sources {
            let (Some(energy), Some(maximum)) = (source.energy_uj, source.max_energy_uj) else {
                continue;
            };
            if !matches!(source.kind.as_str(), "rapl_package" | "rapl_platform") {
                continue;
            }
            if maximum == 0 || energy > maximum {
                continue;
            }
            next_counters.insert(
                source.id.clone(),
                CounterState {
                    energy_uj: energy,
                    max_uj: maximum,
                },
            );
            let Some(previous) = self.open.rapl.get(&source.id) else {
                continue;
            };
            let delta = if energy >= previous.energy_uj {
                energy - previous.energy_uj
            } else if maximum > previous.energy_uj {
                maximum - previous.energy_uj + energy
            } else {
                // The counter lost its maximum or the device disappeared and
                // reappeared with a smaller range; do not credit a guess.
                continue;
            };
            if delta > maximum.max(previous.max_uj) {
                continue;
            }
            rapl_observed = true;
            if !discharging && (source.kind == "rapl_platform" || !platform_counter) {
                rapl_delta_uj = rapl_delta_uj.saturating_add(delta);
            }
        }
        if !discharging && rapl_observed {
            day.cpu_measured = true;
            if rapl_delta_uj > 0 {
                day.cpu_wh = add_wh(day.cpu_wh, rapl_delta_uj as f64 / MICROJOULES_PER_WH);
            }
        }
        let cpu_watts = if !discharging
            && rapl_observed
            && rapl_delta_uj > 0
            && short_interval
            && elapsed.is_some_and(|seconds| seconds > 0.0)
        {
            Some(rapl_delta_uj as f64 / 1_000_000.0 / elapsed.unwrap_or(1.0))
        } else {
            None
        };

        // A RAPL platform domain already includes hwmon components. Package
        // domains include the CPU only, so GPU/storage power remains separate.
        let component_energy_allowed =
            !discharging && !platform_counter && status.components_watts.is_some();
        if component_energy_allowed {
            if let Some(seconds) = elapsed.filter(|_| short_interval) {
                let current = status.components_watts;
                let watts = match (self.open.components_watts, current) {
                    (Some(previous), Some(current)) => (previous + current) / 2.0,
                    (_, Some(current)) => current,
                    _ => 0.0,
                };
                if watts.is_finite() && watts > 0.0 {
                    day.components_wh = add_wh(day.components_wh, watts * seconds / 3600.0);
                }
                day.components_measured = true;
                day.measured_seconds = day
                    .measured_seconds
                    .saturating_add(seconds.round().clamp(0.0, MAX_COMPONENT_GAP_SECONDS) as u64);
            }
        }

        self.open = OpenState {
            date: date.into(),
            last_wall_ms: Some(now_ms),
            ac_online: status.external_power,
            battery_status: status.battery_status.clone(),
            rapl: next_counters,
            battery_wh: battery.and_then(|source| source.energy_wh),
            components_watts: status.components_watts,
        };
        self.last = LastWatts {
            cpu_watts,
            component_watts: status.components_watts,
            battery_watts: if discharging {
                status.battery_watts
            } else {
                None
            },
            at_unix_ms: Some(now_ms),
        };
        self.prune();
    }

    fn prune(&mut self) {
        while self.days.len() > HISTORY_DAYS {
            let Some(oldest) = self.days.keys().next().cloned() else {
                break;
            };
            self.days.remove(&oldest);
        }
        if self.open.rapl.len() > 64 {
            self.open.rapl = BTreeMap::new();
        }
    }

    /// Reject corrupt or implausible persisted values instead of displaying
    /// them as measured energy.
    pub fn sanitize(&mut self) -> bool {
        if self.schema_version != LEDGER_SCHEMA_VERSION || self.machine.is_empty() {
            return false;
        }
        let invalid = self.days.iter().any(|(date, day)| {
            !valid_date(date)
                || [day.cpu_wh, day.components_wh, day.battery_wh]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0 || *value > MAX_DAY_WH)
        });
        self.days.retain(|date, day| {
            valid_date(date)
                && [day.cpu_wh, day.components_wh, day.battery_wh]
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0 && *value <= MAX_DAY_WH)
        });
        for day in self.days.values_mut() {
            // Ledgers written before coverage flags existed cannot distinguish
            // an unavailable part from a measured zero. A nonzero amount is
            // still evidence that the part was measured.
            day.cpu_measured |= day.cpu_wh > 0.0;
            day.components_measured |= day.components_wh > 0.0 || day.measured_seconds > 0;
            day.battery_measured |= day.battery_wh > 0.0;
        }
        self.prune();
        !invalid
    }
}

fn candidate_better(candidate: &LedgerSelection, current: &LedgerSelection) -> bool {
    (
        candidate.fresh,
        candidate.fresh && candidate.origin == "system",
        candidate.ledger.last.at_unix_ms.unwrap_or(0),
    ) > (
        current.fresh,
        current.fresh && current.origin == "system",
        current.ledger.last.at_unix_ms.unwrap_or(0),
    )
}

#[derive(Debug, Clone)]
pub struct LedgerSelection {
    pub ledger: EnergyLedger,
    pub path: PathBuf,
    /// `system` for the optional root sampler, `user` for the desktop ledger.
    pub origin: &'static str,
    pub fresh: bool,
    pub age_seconds: u64,
}

/// Pick the most useful readable ledger without merging two observation
/// streams. A fresh system sampler wins because it can read RAPL counters;
/// otherwise the most recent valid ledger is shown as-is.
pub fn select_ledger(
    machine: &str,
    now: SystemTime,
) -> Result<Option<LedgerSelection>, ControlError> {
    let now_ms = unix_millis(now).unwrap_or(0);
    let shared = PathBuf::from(SHARED_LEDGER_PATH);
    let own = default_ledger_path();
    let mut candidates = Vec::new();
    if own.as_deref() != Some(shared.as_path()) {
        candidates.push((shared, "system"));
    }
    if let Some(own) = own {
        candidates.push((own, "user"));
    }
    let mut best: Option<LedgerSelection> = None;
    for (path, origin) in candidates {
        if !path.exists() {
            continue;
        }
        let ledger = match read_ledger(&path) {
            Ok(ledger) => ledger,
            Err(error) if error.code == "energy_ledger_missing" || origin == "system" => continue,
            Err(_) if best.is_some() => continue,
            Err(error) => return Err(error),
        };
        if ledger.machine != machine {
            continue;
        }
        let age_seconds = now_ms.saturating_sub(ledger.last.at_unix_ms.unwrap_or(0)) / 1000;
        let fresh = ledger.last.at_unix_ms.is_some() && age_seconds <= LEDGER_FRESH_SECONDS;
        let candidate = LedgerSelection {
            ledger,
            path,
            origin,
            fresh,
            age_seconds,
        };
        let replace = best
            .as_ref()
            .is_none_or(|current| candidate_better(&candidate, current));
        if replace {
            best = Some(candidate);
        }
    }
    Ok(best)
}

fn add_wh(value: f64, delta: f64) -> f64 {
    if !delta.is_finite() || delta <= 0.0 {
        return value;
    }
    (value + delta).clamp(0.0, MAX_DAY_WH)
}

pub fn machine_key(identity: &Identity) -> String {
    [
        identity.manufacturer.as_deref(),
        identity.product_name.as_deref(),
        identity.product_version.as_deref(),
        identity.board_name.as_deref(),
        identity.bios_version.as_deref(),
    ]
    .into_iter()
    .map(|value| value.unwrap_or(""))
    .collect::<Vec<_>>()
    .join("|")
}

pub fn unix_millis(now: SystemTime) -> Option<u64> {
    now.duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
}

pub fn unix_now_ms() -> Option<u64> {
    unix_millis(SystemTime::now())
}

/// Local calendar date for the user-visible "today" boundary. Linux uses the
/// process timezone; other platforms fall back to UTC rather than guessing.
pub fn local_date(now: SystemTime) -> String {
    let Ok(duration) = now.duration_since(UNIX_EPOCH) else {
        return utc_date(0);
    };
    let seconds = duration.as_secs().min(i64::MAX as u64) as i64;
    #[cfg(target_os = "linux")]
    {
        let timestamp: libc::time_t = seconds as libc::time_t;
        let mut broken_down: libc::tm = unsafe { std::mem::zeroed() };
        if !unsafe { libc::localtime_r(&timestamp, &mut broken_down) }.is_null() {
            return format!(
                "{:04}-{:02}-{:02}",
                broken_down.tm_year + 1900,
                broken_down.tm_mon + 1,
                broken_down.tm_mday
            );
        }
    }
    utc_date(seconds)
}

fn utc_date(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    if month <= 2 {
        year += 1;
    }
    format!("{year:04}-{month:02}-{day:02}")
}

fn valid_date(date: &str) -> bool {
    let bytes = date.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    if !bytes
        .iter()
        .enumerate()
        .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit())
    {
        return false;
    }
    let month = date[5..7].parse::<u32>().ok();
    let day = date[8..10].parse::<u32>().ok();
    matches!(month, Some(1..=12)) && matches!(day, Some(1..=31))
}

/// Linux sysfs discovery. The testable root is a directory prefix; callers
/// outside tests must pass `/` and cannot select arbitrary files in the CLI.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn status(root: &Path) -> EnergyStatus {
    let mut status = EnergyStatus::default();
    let mut rapl_permission_note = false;
    let mut seen_rapl = HashSet::new();
    for domain in paths(&root.join("class/powercap")) {
        let Some(name) = read(&domain.join("name")) else {
            continue;
        };
        if !seen_rapl.insert(name.clone()) {
            // intel-rapl and intel-rapl-mmio can alias the same domain. Showing
            // both would silently double-charge the same joules.
            status
                .notes
                .push(format!("跳过重复的 RAPL 域 {name}；同名计数器只统计一次。"));
            continue;
        }
        let kind = if name.starts_with("package") {
            "rapl_package"
        } else if name.starts_with("psys") {
            "rapl_platform"
        } else {
            // core/dram/uncore are subdomains of package or platform. Counting
            // them separately would double-charge the same joules.
            continue;
        };
        let energy_path = domain.join("energy_uj");
        let maximum = read(&domain.join("max_energy_range_uj")).and_then(|v| v.parse().ok());
        let energy = read(&energy_path).and_then(|v| v.parse::<u64>().ok());
        let valid =
            matches!((energy, maximum), (Some(value), Some(max)) if max > 0 && value <= max);
        if !valid && energy_path.exists() && energy.is_none() {
            rapl_permission_note = true;
        }
        let device = canonical_device(&domain);
        status.sources.push(PowerSource {
            id: display_path(&energy_path),
            label: format!("RAPL {name}"),
            kind: kind.into(),
            watts: None,
            energy_uj: valid.then_some(energy.unwrap_or(0)),
            max_energy_uj: valid.then_some(maximum.unwrap_or(0)),
            energy_wh: None,
            device_id: device.clone(),
            source: sysfs_source(&energy_path, &device),
            status: if valid {
                CapabilityStatus::Available
            } else {
                CapabilityStatus::Unavailable
            },
            reason: (!valid).then(|| {
                "RAPL energy_uj is missing, invalid or readable only by root; CPU package energy is not counted. Install the optional system sampler for counter-based CPU energy.".into()
            }),
        });
    }
    if rapl_permission_note {
        status.notes.push(
            "CPU 封装能量计数器 energy_uj 仅 root 可读；普通用户会话不会伪造 CPU 功耗。".into(),
        );
    }

    let battery_names: HashSet<String> = paths(&root.join("class/power_supply"))
        .into_iter()
        .filter(|supply| read(&supply.join("type")).as_deref() == Some("Battery"))
        .filter_map(|supply| {
            supply
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect();
    let mut components = 0.0;
    let mut component_measured = false;
    for hwmon in paths(&root.join("class/hwmon")) {
        let chip = read(&hwmon.join("name")).unwrap_or_else(|| "unknown hwmon".into());
        if battery_names
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&chip))
        {
            // A battery hwmon power channel is charger input or battery flow,
            // not a component of system consumption. The battery source below
            // owns that measurement.
            continue;
        }
        let device = canonical_device(&hwmon);
        let suspended =
            read(&hwmon.join("device/power/runtime_status")).as_deref() == Some("suspended");
        for input in paths(&hwmon) {
            let name = input
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let Some(channel) = power_channel(&name) else {
                continue;
            };
            let label = read(&hwmon.join(format!("{channel}_label")))
                .map(|label| format!("{chip}: {label}"))
                .unwrap_or_else(|| format!("{chip}: {channel}"));
            let value = read(&input)
                .and_then(|v| v.parse::<f64>().ok())
                .map(|micro| micro / 1_000_000.0)
                .filter(|watts| watts.is_finite() && (0.0..=MAX_POWER_WATTS).contains(watts));
            let available = value.is_some() && !suspended;
            if available {
                components += value.unwrap_or(0.0);
                component_measured = true;
            }
            status.sources.push(PowerSource {
                id: format!("{}/{}", display_path(&hwmon), name),
                label,
                kind: "hwmon_power".into(),
                watts: available.then_some(value.unwrap_or(0.0)),
                energy_uj: None,
                max_energy_uj: None,
                energy_wh: None,
                device_id: device.clone(),
                source: sysfs_source(&input, &device),
                status: if available {
                    CapabilityStatus::Available
                } else {
                    CapabilityStatus::Unavailable
                },
                reason: (!available).then(|| {
                    if suspended {
                        "Device is runtime-suspended; power was not read to avoid waking it".into()
                    } else {
                        "Missing, unreadable or invalid hwmon power input; not reported as zero"
                            .into()
                    }
                }),
            });
        }
    }
    if component_measured {
        status.components_watts = Some(round_watts(components));
    }

    for supply in paths(&root.join("class/power_supply")) {
        let device = canonical_device(&supply);
        let name = supply
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        match read(&supply.join("type")).as_deref() {
            Some("Battery") => {
                let battery_status = read(&supply.join("status"));
                let discharging = battery_status
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case("Discharging"));
                let watts = if discharging {
                    battery_watts(&supply)
                } else {
                    None
                };
                let energy_wh = battery_energy_wh(&supply);
                let available = discharging && (watts.is_some() || energy_wh.is_some());
                if discharging {
                    status.battery_watts = watts;
                }
                status.battery_status = battery_status.clone();
                let power_path = if supply.join("power_now").exists() {
                    supply.join("power_now")
                } else {
                    supply.join("current_now")
                };
                status.sources.push(PowerSource {
                    id: format!("{}/battery_discharge", display_path(&supply)),
                    label: format!("{name}: battery discharge"),
                    kind: "battery_discharge".into(),
                    watts: if available { watts } else { None },
                    energy_uj: None,
                    max_energy_uj: None,
                    energy_wh,
                    device_id: device.clone(),
                    source: sysfs_source(&power_path, &device),
                    status: if available {
                        CapabilityStatus::Available
                    } else {
                        CapabilityStatus::Unavailable
                    },
                    reason: (!available).then(|| {
                        format!(
                            "Battery status is {}; battery power is only system consumption while discharging, not charger input",
                            battery_status.as_deref().unwrap_or("unknown")
                        )
                    }),
                });
            }
            Some("Mains" | "USB" | "USB_C" | "USB_PD") => {
                let online = read(&supply.join("online"));
                if online.as_deref() == Some("1") {
                    status.external_power = Some(true);
                } else if online.as_deref() == Some("0") && status.external_power.is_none() {
                    status.external_power = Some(false);
                }
            }
            _ => {}
        }
    }

    if status.battery_watts.is_none() {
        status.notes.push(
            "当前没有电池放电功率，无法得到整机功耗（AC 供电时输入功率和适配器损耗不可得）。"
                .into(),
        );
    }
    if status.components_watts.is_some() {
        status
            .notes
            .push("部件功耗是 hwmon 瞬时读数之和，不含屏幕、内存、硬盘、风扇与供电损耗。".into());
    }
    status
}

#[cfg(any(target_os = "linux", test))]
fn battery_watts(supply: &Path) -> Option<f64> {
    if let Some(watts) = read(&supply.join("power_now")).and_then(|v| v.parse::<f64>().ok()) {
        let watts = watts / 1_000_000.0;
        if watts.is_finite() && (0.0..=MAX_POWER_WATTS).contains(&watts) {
            return Some(watts);
        }
    }
    let current = read(&supply.join("current_now"))?.parse::<f64>().ok()?;
    let voltage = read(&supply.join("voltage_now"))?.parse::<f64>().ok()?;
    let watts = current * voltage / 1e12;
    (watts.is_finite() && (0.0..=MAX_POWER_WATTS).contains(&watts)).then_some(watts)
}

#[cfg(any(target_os = "linux", test))]
fn battery_energy_wh(supply: &Path) -> Option<f64> {
    if let Some(energy) = read(&supply.join("energy_now")).and_then(|v| v.parse::<f64>().ok()) {
        let watthours = energy / 1_000_000.0;
        if watthours.is_finite() && (0.0..=MAX_DAY_WH).contains(&watthours) {
            return Some(watthours);
        }
    }
    let charge = read(&supply.join("charge_now"))?.parse::<f64>().ok()?;
    let voltage = read(&supply.join("voltage_now"))?.parse::<f64>().ok()?;
    let watthours = charge * voltage / 1e12;
    (watthours.is_finite() && (0.0..=MAX_DAY_WH).contains(&watthours)).then_some(watthours)
}

#[cfg(any(target_os = "linux", test))]
fn round_watts(watts: f64) -> f64 {
    (watts * 1000.0).round() / 1000.0
}

#[cfg(any(target_os = "linux", test))]
fn power_channel(name: &str) -> Option<String> {
    let stem = name.strip_suffix("_input")?;
    let digits = stem.strip_prefix("power")?;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(stem.to_owned())
}

#[cfg(any(target_os = "linux", test))]
fn paths(root: &Path) -> Vec<PathBuf> {
    let mut entries = fs::read_dir(root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .take(256)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

#[cfg(any(target_os = "linux", test))]
fn read(path: &Path) -> Option<String> {
    let mut value = String::new();
    File::open(path)
        .ok()?
        .take(4097)
        .read_to_string(&mut value)
        .ok()?;
    if value.len() > 4096 {
        return None;
    }
    let value = value.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    (!value.is_empty()).then(|| value.to_owned())
}

#[cfg(any(target_os = "linux", test))]
fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(any(target_os = "linux", test))]
fn canonical_device(path: &Path) -> String {
    let resolved = fs::canonicalize(path.join("device"))
        .or_else(|_| fs::canonicalize(path))
        .unwrap_or_else(|_| path.into());
    display_path(&resolved)
}

#[cfg(any(target_os = "linux", test))]
fn sysfs_source(path: &Path, device: &str) -> EvidenceSource {
    EvidenceSource {
        backend: "linux_sysfs".into(),
        path: display_path(path),
        device_id: Some(device.into()),
    }
}

/// Default writable ledger. Root owns the shared file used by the optional
/// system sampler; a desktop session keeps its own state under XDG_STATE_HOME.
pub fn default_ledger_path() -> Option<PathBuf> {
    #[cfg(unix)]
    if unsafe { libc::geteuid() } == 0 {
        return Some(PathBuf::from(SHARED_LEDGER_PATH));
    }
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local/state"))
        })?;
    Some(base.join("mechrevo-console/energy.json"))
}

pub fn read_ledger(path: &Path) -> Result<EnergyLedger, ControlError> {
    let mut file = open_read(path)?;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_LEDGER_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| ControlError::new("energy_ledger_read_failed", error.to_string()))?;
    if bytes.len() as u64 > MAX_LEDGER_BYTES {
        return Err(ControlError::new(
            "energy_ledger_invalid",
            "Energy ledger exceeds the 64 KiB read bound",
        ));
    }
    let mut ledger: EnergyLedger = serde_json::from_slice(&bytes)
        .map_err(|error| ControlError::new("energy_ledger_invalid", error.to_string()))?;
    if !ledger.sanitize() {
        return Err(ControlError::new(
            "energy_ledger_invalid",
            "Energy ledger schema or machine identity is invalid",
        ));
    }
    Ok(ledger)
}

/// Read-modify-write under an advisory lock. Used by CLI samples and the GUI's
/// periodic flush so a timer cannot silently overwrite another observation.
pub fn sample_ledger(
    path: &Path,
    machine: &str,
    status: &EnergyStatus,
    now: SystemTime,
) -> Result<EnergyLedger, ControlError> {
    with_lock(path, || {
        let mut ledger = match read_ledger(path) {
            Ok(ledger) => ledger,
            Err(error)
                if matches!(
                    error.code.as_str(),
                    "energy_ledger_missing" | "energy_ledger_read_failed" | "energy_ledger_invalid"
                ) =>
            {
                EnergyLedger::new(machine)
            }
            Err(error) => return Err(error),
        };
        ledger.observe(status, machine, now);
        write_ledger(path, &ledger)?;
        Ok(ledger)
    })
}

pub fn write_ledger(path: &Path, ledger: &EnergyLedger) -> Result<(), ControlError> {
    let parent = path.parent().ok_or_else(|| {
        ControlError::new("energy_ledger_write_failed", "Ledger path has no parent")
    })?;
    fs::create_dir_all(parent)
        .map_err(|error| ControlError::new("energy_ledger_write_failed", error.to_string()))?;
    let json = serde_json::to_vec_pretty(ledger)
        .map_err(|error| ControlError::new("energy_ledger_write_failed", error.to_string()))?;
    if json.len() as u64 > MAX_LEDGER_BYTES {
        return Err(ControlError::new(
            "energy_ledger_write_failed",
            "Energy ledger exceeds the 64 KiB write bound",
        ));
    }
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.subsec_nanos())
            .unwrap_or(0)
    ));
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        set_mode(&mut options);
        let mut file = options.open(&temporary)?;
        file.write_all(&json)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok::<(), std::io::Error>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|error| ControlError::new("energy_ledger_write_failed", error.to_string()))
}

fn with_lock<T>(
    path: &Path,
    operation: impl FnOnce() -> Result<T, ControlError>,
) -> Result<T, ControlError> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    {
        let parent = path.parent().ok_or_else(|| {
            ControlError::new("energy_ledger_lock_failed", "Ledger path has no parent")
        })?;
        fs::create_dir_all(parent)
            .map_err(|error| ControlError::new("energy_ledger_lock_failed", error.to_string()))?;
        let lock_path = parent.join(format!(
            ".{}.lock",
            path.file_name().unwrap_or_default().to_string_lossy()
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options
            .open(&lock_path)
            .map_err(|error| ControlError::new("energy_ledger_lock_failed", error.to_string()))?;
        use std::os::fd::AsRawFd;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(ControlError::new(
                "energy_ledger_lock_failed",
                std::io::Error::last_os_error().to_string(),
            ));
        }
        let result = operation();
        let _ = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        result
    }
    #[cfg(not(unix))]
    operation()
}

fn open_read(path: &Path) -> Result<File, ControlError> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    options.open(path).map_err(|error| {
        let code = if error.kind() == std::io::ErrorKind::NotFound {
            "energy_ledger_missing"
        } else {
            "energy_ledger_read_failed"
        };
        ControlError::new(code, error.to_string())
    })
}

fn set_mode(options: &mut OpenOptions) {
    #[cfg(not(unix))]
    let _ = options;
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mode = unsafe {
            if libc::geteuid() == 0 {
                0o644
            } else {
                0o600
            }
        };
        options.mode(mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rapl(kind: &str, id: &str, energy_uj: u64, maximum: u64) -> PowerSource {
        PowerSource {
            id: id.into(),
            label: id.into(),
            kind: kind.into(),
            watts: None,
            energy_uj: Some(energy_uj),
            max_energy_uj: Some(maximum),
            energy_wh: None,
            device_id: "test".into(),
            source: EvidenceSource {
                backend: "test".into(),
                path: id.into(),
                device_id: None,
            },
            status: CapabilityStatus::Available,
            reason: None,
        }
    }

    fn components(watts: f64) -> EnergyStatus {
        EnergyStatus {
            components_watts: Some(watts),
            ..EnergyStatus::default()
        }
    }

    #[test]
    fn utc_civil_dates_include_leap_day_and_epoch() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(951_782_400), "2000-02-29");
        assert_eq!(utc_date(1_700_000_000), "2023-11-14");
    }

    #[test]
    fn ledger_integrates_short_component_intervals_only() {
        let status = components(10.0);
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&status, "machine", "2026-11-10", 1_000);
        ledger.observe_on(&status, "machine", "2026-11-10", 4_000);
        assert!(
            (ledger.today("2026-11-10").unwrap().components_wh - 10.0 * 3.0 / 3600.0).abs() < 1e-9
        );
        assert_eq!(ledger.today("2026-11-10").unwrap().measured_seconds, 3);
        ledger.observe_on(&status, "machine", "2026-11-10", 7_000);
        let day = ledger.today("2026-11-10").unwrap();
        assert!((day.components_wh - 10.0 * 6.0 / 3600.0).abs() < 1e-9);
        assert_eq!(day.measured_seconds, 6);
        assert!(day.components_measured);
        assert!(!day.cpu_measured);
        assert!(day.has_measurement());
        let before = day.components_wh;
        ledger.observe_on(&status, "machine", "2026-11-10", 100_000);
        assert_eq!(ledger.today("2026-11-10").unwrap().components_wh, before);
    }

    #[test]
    fn rapl_counter_is_exact_across_long_gaps_and_wraps() {
        let mut status = EnergyStatus {
            sources: vec![rapl("rapl_package", "pkg", 3_000_000_000, 4_000_000_000)],
            ..EnergyStatus::default()
        };
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&status, "machine", "2026-11-10", 1_000);
        status.sources[0].energy_uj = Some(3_600_000_000);
        ledger.observe_on(&status, "machine", "2026-11-10", 7_200_000);
        let day = ledger.today("2026-11-10").unwrap();
        assert!((day.cpu_wh - 1.0 / 6.0).abs() < 1e-9);
        assert!(day.cpu_measured);
        assert_eq!(ledger.last.cpu_watts, None, "long gaps do not become watts");
        // Wrap from 3.6 GJ to 0.2 GJ using a 4 GJ maximum = 0.6 GJ.
        status.sources[0].energy_uj = Some(200_000_000);
        ledger.observe_on(&status, "machine", "2026-11-10", 7_260_000);
        let day = ledger.today("2026-11-10").unwrap();
        assert!((day.cpu_wh - (1.0 / 6.0 + 0.6 / 3.6)).abs() < 1e-9);
    }

    #[test]
    fn subdomain_counters_are_not_double_counted() {
        let mut status = EnergyStatus {
            sources: vec![
                rapl("rapl_package", "pkg", 1_000_000_000, 4_000_000_000),
                rapl("rapl_subdomain", "core", 900_000_000, 4_000_000_000),
            ],
            ..EnergyStatus::default()
        };
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&status, "machine", "2026-11-10", 1_000);
        status.sources[0].energy_uj = Some(2_800_000_000);
        status.sources[1].energy_uj = Some(2_700_000_000);
        ledger.observe_on(&status, "machine", "2026-11-10", 4_000);
        assert!((ledger.today("2026-11-10").unwrap().cpu_wh - 0.5).abs() < 1e-9);
    }

    #[test]
    fn battery_discharge_covers_the_interval_and_suppresses_components() {
        let mut status = EnergyStatus {
            external_power: Some(false),
            battery_status: Some("Discharging".into()),
            battery_watts: Some(20.0),
            components_watts: Some(8.0),
            ..EnergyStatus::default()
        };
        status.sources = vec![PowerSource {
            id: "bat".into(),
            label: "BAT0".into(),
            kind: "battery_discharge".into(),
            watts: Some(20.0),
            energy_uj: None,
            max_energy_uj: None,
            energy_wh: Some(40.0),
            device_id: "bat".into(),
            source: EvidenceSource {
                backend: "test".into(),
                path: "bat".into(),
                device_id: None,
            },
            status: CapabilityStatus::Available,
            reason: None,
        }];
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&status, "machine", "2026-11-10", 1_000);
        status.sources[0].energy_wh = Some(39.0);
        ledger.observe_on(&status, "machine", "2026-11-10", 7_200_000);
        let day = ledger.today("2026-11-10").unwrap();
        assert_eq!(day.battery_wh, 1.0);
        assert_eq!(day.components_wh, 0.0);
        assert_eq!(day.cpu_wh, 0.0);
        assert!(day.battery_measured);
        assert!(!day.cpu_measured);
        assert!(!day.components_measured);
    }

    #[test]
    fn identity_change_and_day_rollover_reset_baselines() {
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&components(10.0), "machine", "2026-11-10", 1_000);
        ledger.observe_on(&components(10.0), "machine", "2026-11-10", 4_000);
        ledger.observe_on(&components(10.0), "machine", "2026-11-11", 7_000);
        assert_eq!(
            ledger.today("2026-11-11").unwrap().components_wh,
            0.0,
            "the first sample of a new day is a baseline"
        );
        ledger.observe_on(&components(10.0), "other", "2026-11-11", 10_000);
        assert_eq!(ledger.machine, "other");
        assert!(ledger.today("2026-11-10").is_none());
    }

    #[test]
    fn corrupt_ledger_values_are_rejected() {
        let mut ledger = EnergyLedger::new("machine");
        ledger.days.insert(
            "2026-11-10".into(),
            EnergyDay {
                cpu_wh: f64::NAN,
                ..EnergyDay::default()
            },
        );
        assert!(!ledger.sanitize());
        assert!(ledger.days.is_empty());
        let mut ledger = EnergyLedger::new("machine");
        ledger.days.insert(
            "not-a-date".into(),
            EnergyDay {
                cpu_wh: 1.0,
                ..EnergyDay::default()
            },
        );
        assert!(!ledger.sanitize());
        assert!(ledger.days.is_empty());
    }

    #[test]
    fn date_validation_is_strict() {
        assert!(valid_date("2026-11-10"));
        assert!(!valid_date("2026-13-10"));
        assert!(!valid_date("2026-11-32"));
        assert!(!valid_date("2026-1-10"));
        assert!(!valid_date("2026/11/10"));
    }

    fn selection(machine: &str, fresh: bool, at: u64, origin: &'static str) -> LedgerSelection {
        let mut ledger = EnergyLedger::new(machine);
        ledger.last.at_unix_ms = Some(at);
        LedgerSelection {
            ledger,
            path: PathBuf::from(machine),
            origin,
            fresh,
            age_seconds: 0,
        }
    }

    #[test]
    fn fresh_system_sampler_wins_but_a_newer_ledger_breaks_a_stale_tie() {
        let stale_system = selection("m", false, 10, "system");
        let fresh_user = selection("m", true, 20, "user");
        let stale_user = selection("m", false, 30, "user");
        let fresh_system = selection("m", true, 5, "system");
        assert!(candidate_better(&fresh_user, &stale_system));
        assert!(candidate_better(&stale_user, &stale_system));
        assert!(candidate_better(&fresh_system, &fresh_user));
        assert!(!candidate_better(&stale_system, &stale_user));
    }

    #[test]
    fn microjoules_are_converted_at_the_wh_boundary() {
        assert!((3_600_000_000_f64 / MICROJOULES_PER_WH - 1.0).abs() < f64::EPSILON);
    }

    fn put(root: &Path, relative: &str, value: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, value).unwrap();
    }

    #[test]
    fn sysfs_power_sources_keep_scope_and_unknown_values() {
        let root = std::env::temp_dir().join(format!(
            "mechrevo-energy-sysfs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        put(&root, "class/hwmon/hwmon0/name", "amdgpu");
        put(&root, "class/hwmon/hwmon0/power1_input", "50804000");
        put(&root, "class/hwmon/hwmon0/power1_label", "PPT");
        put(&root, "class/hwmon/hwmon1/name", "nvme");
        put(&root, "class/hwmon/hwmon1/power1_input", "invalid");
        put(&root, "class/hwmon/hwmon2/name", "BAT0");
        put(&root, "class/hwmon/hwmon2/power1_input", "1000000");
        put(&root, "class/powercap/intel-rapl:0/name", "package-0");
        put(&root, "class/powercap/intel-rapl:0/energy_uj", "1000000");
        put(
            &root,
            "class/powercap/intel-rapl:0/max_energy_range_uj",
            "4000000000",
        );
        put(&root, "class/powercap/intel-rapl:0:0/name", "core");
        put(&root, "class/powercap/intel-rapl:0:0/energy_uj", "900000");
        put(
            &root,
            "class/powercap/intel-rapl:0:0/max_energy_range_uj",
            "4000000000",
        );
        put(&root, "class/power_supply/BAT0/type", "Battery");
        put(&root, "class/power_supply/BAT0/status", "Discharging");
        put(&root, "class/power_supply/BAT0/current_now", "2000000");
        put(&root, "class/power_supply/BAT0/voltage_now", "16000000");
        put(&root, "class/power_supply/BAT0/charge_now", "2800000");
        put(&root, "class/power_supply/AC0/type", "Mains");
        put(&root, "class/power_supply/AC0/online", "0");
        let status = status(&root);
        assert_eq!(status.external_power, Some(false));
        assert_eq!(
            status.components_watts,
            Some(50.804),
            "battery charger input must not be summed as a component"
        );
        assert_eq!(status.battery_watts, Some(32.0));
        assert_eq!(status.battery_status.as_deref(), Some("Discharging"));
        let rapl: Vec<_> = status
            .sources
            .iter()
            .filter(|source| source.kind.starts_with("rapl_"))
            .collect();
        assert_eq!(rapl.len(), 1, "core is a subdomain and must not be counted");
        assert_eq!(rapl[0].energy_uj, Some(1_000_000));
        let unknown = status
            .sources
            .iter()
            .find(|source| source.label.contains("nvme"))
            .unwrap();
        assert_eq!(unknown.status, CapabilityStatus::Unavailable);
        assert_eq!(unknown.watts, None);
        let battery = status
            .sources
            .iter()
            .find(|source| source.kind == "battery_discharge")
            .unwrap();
        assert!((battery.energy_wh.unwrap() - 44.8).abs() < 1e-9);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn full_or_charging_battery_is_not_system_consumption() {
        let root = std::env::temp_dir().join(format!(
            "mechrevo-energy-full-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        put(&root, "class/power_supply/BAT0/type", "Battery");
        put(&root, "class/power_supply/BAT0/status", "Full");
        put(&root, "class/power_supply/BAT0/current_now", "0");
        put(&root, "class/power_supply/BAT0/voltage_now", "16114000");
        let status = status(&root);
        assert_eq!(status.battery_watts, None);
        let battery = status
            .sources
            .iter()
            .find(|source| source.kind == "battery_discharge")
            .unwrap();
        assert_eq!(battery.status, CapabilityStatus::Unavailable);
        assert!(battery.reason.as_deref().unwrap().contains("Full"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn ledger_round_trip_and_size_bound() {
        let directory = std::env::temp_dir().join(format!(
            "mechrevo-energy-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("energy.json");
        let mut ledger = EnergyLedger::new("machine");
        ledger.observe_on(&components(12.5), "machine", "1999-01-01", 1_000);
        ledger.observe_on(&components(12.5), "machine", "1999-01-01", 4_000);
        write_ledger(&path, &ledger).unwrap();
        let loaded = read_ledger(&path).unwrap();
        assert_eq!(loaded.machine, "machine");
        assert!(
            (loaded.today("1999-01-01").unwrap().components_wh - 12.5 * 3.0 / 3600.0).abs() < 1e-9
        );
        let now = SystemTime::now();
        let date = local_date(now);
        let sampled = sample_ledger(&path, "machine", &components(12.5), now).unwrap();
        assert_eq!(sampled.today(&date).unwrap().samples, 1);
        assert_eq!(sampled.machine, "machine");
        fs::remove_dir_all(&directory).unwrap();
    }
}
