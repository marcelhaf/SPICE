/// Global matrix type used by all elements.
///
/// The matrix has `num_vars + 1` rows and `num_vars + 2` columns:
/// the last column is the right‑hand side (independent sources).
pub type Matrix = Vec<Vec<f64>>;

/// Constant π used in sinusoidal sources.
const PI: f64 = 3.14159265358979323846;

/// Floating–point version of `fmod`, used for periodic waveforms.
///
/// # Arguments
/// * `a` - Value to be wrapped.
/// * `p` - Period. If zero, returns 0 to avoid division by zero.
///
/// # Returns
/// Remainder of `a / p` in the range \([0, p)\) if `p > 0`.
fn fmod_like(a: f64, p: f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    let k = (a / p).trunc();
    a - k * p
}

/// Common data stored by all circuit elements.
///
/// Node indices are 0‑based, with node 0 representing ground.
/// `jx` and `jy` are auxiliary branch current variables used by
/// voltage sources, inductors, controlled sources, etc.
#[derive(Clone, Debug)]
pub struct ElementBase {
    pub name: String,
    pub na: usize,
    pub nb: usize,
    pub nc: usize,
    pub nd: usize,
    pub jx: usize,
    pub jy: usize,
    pub value: f64,
}

impl ElementBase {
    /// Creates a new base with a name and two terminal nodes.
    ///
    /// All other fields are initialized to zero.
    pub fn new(name: String, na: usize, nb: usize) -> Self {
        Self {
            name,
            na,
            nb,
            nc: 0,
            nd: 0,
            jx: 0,
            jy: 0,
            value: 0.0,
        }
    }
}

/// Trait implemented by every circuit element.
///
/// The solver calls `add_stamp` to let each element contribute its
/// entries to the Modified Nodal Analysis (MNA) matrix.
pub trait Element {
    /// Returns an immutable reference to the common base fields.
    fn base(&self) -> &ElementBase;

    /// Returns a mutable reference to the common base fields.
    fn base_mut(&mut self) -> &mut ElementBase;

    /// Adds this element's stamp to the global system matrix.
    ///
    /// # Arguments
    /// * `system`      - Global MNA matrix to be modified in place.
    /// * `time`        - Current simulation time.
    /// * `dt`          - Time step size.
    /// * `sol_time`    - Solution at the previous time step
    ///                   (used by dynamic elements).
    /// * `sol_nr`      - Current Newton–Raphson solution estimate.
    /// * `sol_prev_nr` - Previous Newton–Raphson solution.
    /// * `converged`   - Convergence flags for each variable.
    /// * `gmin`        - Global gmin used in gmin stepping.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        time: f64,
        dt: f64,
        sol_time: &[f64],
        sol_nr: &[f64],
        sol_prev_nr: &[f64],
        converged: &[bool],
        gmin: f64,
    );
}

// ========================= Resistor (R) =========================

/// Linear resistor between nodes `na` and `nb`.
pub struct Resistor {
    base: ElementBase,
}

impl Resistor {
    /// Creates a resistor with resistance `r` (ohms).
    pub fn new(name: String, a: usize, b: usize, r: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.value = r;
        Self { base }
    }
}

impl Element for Resistor {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the conductance \(g = 1/R\) between `na` and `nb`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let g = 1.0 / self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;

        system[na][na] += g;
        system[nb][nb] += g;
        system[na][nb] -= g;
        system[nb][na] -= g;
    }
}

// =========== Transconductance (G) – VCCS ===========

/// Voltage‑controlled current source (VCCS).
///
/// Injects a current proportional to the voltage between `nc` and `nd`.
pub struct Transconductance {
    base: ElementBase,
}

impl Transconductance {
    /// Creates a VCCS with transconductance `g_val` (A/V).
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize, g_val: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = g_val;
        Self { base }
    }
}

impl Element for Transconductance {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the controlled current:
    /// \(I_{ab} = g (V_c - V_d)\).
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let g = self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;

        system[na][nc] += g;
        system[nb][nd] += g;
        system[na][nd] -= g;
        system[nb][nc] -= g;
    }
}

// ===== VCVS (E) – Voltage controlled voltage source =====

/// Voltage‑controlled voltage source (VCVS).
///
/// Output voltage between `na` and `nb` is `gain * (V_nc - V_nd)`.
pub struct VoltageControlledVoltageSource {
    base: ElementBase,
}

impl VoltageControlledVoltageSource {
    /// Creates a VCVS with given gain.
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize, gain: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = gain;
        Self { base }
    }
}

impl Element for VoltageControlledVoltageSource {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the VCVS using an auxiliary current variable `jx`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let g = self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;
        let jx = self.base.jx;

        system[na][jx] += 1.0;
        system[nb][jx] -= 1.0;
        system[jx][na] -= 1.0;
        system[jx][nb] += 1.0;
        system[jx][nc] += g;
        system[jx][nd] -= g;
    }
}

// ===== CCCS (F) – Current controlled current source =====

/// Current‑controlled current source (CCCS).
///
/// Output current between `na` and `nb` is `gain * I(jx)` measured
/// through the controlling element.
pub struct CurrentControlledCurrentSource {
    base: ElementBase,
}

impl CurrentControlledCurrentSource {
    /// Creates a CCCS with gain `gain` (dimensionless).
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize, gain: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = gain;
        Self { base }
    }
}

impl Element for CurrentControlledCurrentSource {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the CCCS using the auxiliary current variable `jx`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let g = self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;
        let jx = self.base.jx;

        system[na][jx] += g;
        system[nb][jx] -= g;
        system[nc][jx] += 1.0;
        system[nd][jx] -= 1.0;
        system[jx][nc] -= 1.0;
        system[jx][nd] += 1.0;
    }
}

// ===== CCVS (H) – Transresistance =====

/// Current‑controlled voltage source (CCVS).
///
/// Output voltage is `r_val * I(control_branch)`.
pub struct Transresistance {
    base: ElementBase,
}

impl Transresistance {
    /// Creates a CCVS with transresistance `r_val` (ohms).
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize, r_val: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = r_val;
        Self { base }
    }
}

impl Element for Transresistance {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the CCVS using two auxiliary currents `jx` and `jy`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let g = self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;
        let jx = self.base.jx;
        let jy = self.base.jy;

        system[na][jy] += 1.0;
        system[nb][jy] -= 1.0;
        system[nc][jx] += 1.0;
        system[nd][jx] -= 1.0;
        system[jy][na] -= 1.0;
        system[jy][nb] += 1.0;
        system[jx][nc] -= 1.0;
        system[jx][nd] += 1.0;
        system[jy][jx] += g;
    }
}

// ===== Ideal op‑amp (O) =====

/// Ideal operational amplifier.
///
/// Enforces \(V_{na} - V_{nb} = V_{nc} - V_{nd}\) using a large gain.
pub struct OpAmp {
    base: ElementBase,
}

impl OpAmp {
    /// Creates an ideal op‑amp with output nodes `a`, `b`
    /// and input nodes `c`, `d`.
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        Self { base }
    }
}

impl Element for OpAmp {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the ideal op‑amp constraint using an auxiliary current `jx`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;
        let jx = self.base.jx;

        system[na][jx] += 1.0;
        system[nb][jx] -= 1.0;
        system[jx][nc] += 1.0;
        system[jx][nd] -= 1.0;
    }
}

// ===== Ideal transformer (K) =====

/// Ideal transformer with turns ratio `n`.
pub struct IdealTransformer {
    base: ElementBase,
}

impl IdealTransformer {
    /// Creates an ideal transformer with turns ratio `n`.
    pub fn new(name: String, a: usize, b: usize, c: usize, d: usize, n: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = n;
        Self { base }
    }
}

impl Element for IdealTransformer {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the ideal transformer relations between primary and secondary.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let n = self.base.value;
        let na = self.base.na;
        let nb = self.base.nb;
        let nc = self.base.nc;
        let nd = self.base.nd;
        let jx = self.base.jx;

        system[jx][na] += n;
        system[jx][nb] -= n;
        system[jx][nc] -= 1.0;
        system[jx][nd] += 1.0;
        system[na][jx] -= n;
        system[nb][jx] += n;
        system[nc][jx] += 1.0;
        system[nd][jx] -= 1.0;
    }
}

// ========== Generic current source (I) ==========

/// Independent current source with DC, SIN or PULSE waveform.
pub struct CurrentSource {
    base: ElementBase,

    kind: String,
    params: Vec<String>,

    // SIN parameters
    a0: f64,
    a: f64,
    f: f64,
    ta: f64,
    alpha: f64,
    k_deg: f64,
    c: f64,

    // PULSE parameters
    k2: f64,
    a2: f64,
    ta2: f64,
    ts: f64,
    td: f64,
    t1: f64,
    p: f64,
    c2: f64,
}

impl CurrentSource {
    /// Creates a current source from a list of string parameters.
    ///
    /// `params[0]` is the kind: `"DC"`, `"SIN"` or `"PULSE"`.
    pub fn new(name: String, a: usize, b: usize, params: Vec<String>) -> Self {
        let mut base = ElementBase::new(name, a, b);

        let kind = params.get(0).cloned().unwrap_or_default();
        base.value = 0.0;

        let mut me = Self {
            base,
            kind,
            params,
            a0: 0.0,
            a: 0.0,
            f: 0.0,
            ta: 0.0,
            alpha: 0.0,
            k_deg: 0.0,
            c: 0.0,
            k2: 0.0,
            a2: 0.0,
            ta2: 0.0,
            ts: 0.0,
            td: 0.0,
            t1: 0.0,
            p: 0.0,
            c2: 0.0,
        };

        me.set_param();
        me
    }

    /// Parses the parameter list for SIN or PULSE waveforms.
    fn set_param(&mut self) {
        if self.kind == "SIN" {
            self.a0 = self.params[1].parse().unwrap();
            self.a = self.params[2].parse().unwrap();
            self.f = self.params[3].parse().unwrap();
            self.ta = self.params[4].parse().unwrap();
            self.alpha = self.params[5].parse().unwrap();
            self.k_deg = self.params[6].parse().unwrap();
            self.c = self.params[7].parse().unwrap();
        } else if self.kind == "PULSE" {
            if self.params.len() < 8 {
                panic!("PULSE requires 7 numeric parameters");
            }
            self.k2 = self.params[1].parse().expect("invalid k2");
            self.a2 = self.params[2].parse().expect("invalid A2");
            self.ta2 = self.params[3].parse().expect("invalid ta2");
            self.ts = self.params[4].parse().expect("invalid ts");
            self.td = self.params[5].parse().expect("invalid td");
            self.t1 = self.params[6].parse().expect("invalid t1");
            self.p = self.params[7].parse().expect("invalid p");
        }
    }

    /// Updates `base.value` according to the waveform at a given time.
    ///
    /// For SIN, the expression is:
    /// \[
    /// I(t) = A_0 + A e^{-\alpha (t - t_a)}
    ///        \sin(2 \pi f (t - t_a) + \pi k / 180)
    /// \]
    fn set_value(&mut self, time: f64, dt: f64) {
        if self.kind == "DC" {
            self.base.value = self.params[1].parse().unwrap();
            return;
        }

        if self.kind == "SIN" {
            let tf = (self.c / self.f) + self.ta;
            let mut t = time;

            if time <= self.ta {
                t = self.ta;
            } else if time > tf {
                t = tf;
            }

            self.base.value = self.a0
                + self.a
                    * (-self.alpha * (t - self.ta)).exp()
                    * (2.0 * PI * self.f * (t - self.ta) + PI * self.k_deg / 180.0).sin();
            return;
        }

        if self.kind == "PULSE" {
            let tf = (self.c2 * self.p) + self.ta2;
            let mut t = fmod_like(time - self.ta2, self.p);

            let mut ts = self.ts;
            let mut td = self.td;

            if ts == 0.0 {
                ts = dt;
            }
            if td == 0.0 {
                td = dt;
            }

            if time > self.ta2 && time <= tf {
                if t >= 0.0 && t < ts {
                    let m = (self.a2 - self.k2) / ts;
                    let b = self.k2;
                    self.base.value = m * t + b;
                } else if t >= ts && t <= (ts + self.t1) {
                    self.base.value = self.a2;
                } else if t > (ts + self.t1) && t <= (ts + self.t1 + td) {
                    let m = (self.k2 - self.a2) / td;
                    let b = -m * (ts + self.t1) + self.a2;
                    self.base.value = m * t + b;
                } else if t > (ts + self.t1 + td) {
                    self.base.value = self.k2;
                }
            } else {
                self.base.value = self.k2;
            }
        }
    }
}

impl Element for CurrentSource {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the independent current into the right‑hand side.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        time: f64,
        dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let na = self.base.na;
        let nb = self.base.nb;
        let s = system[0].len() - 1;

        self.set_value(time, dt);

        let i = self.base.value;
        system[na][s] += i;
        system[nb][s] -= i;
    }
}

// ========== Generic voltage source (V) ==========

/// Independent voltage source with DC, SIN or PULSE waveform.
pub struct VoltageSource {
    base: ElementBase,

    kind: String,
    params: Vec<String>,

    // SIN parameters
    a0: f64,
    a: f64,
    f: f64,
    ta: f64,
    alpha: f64,
    k_deg: f64,
    c: f64,

    // PULSE parameters
    k2: f64,
    a2: f64,
    ta2: f64,
    ts: f64,
    td: f64,
    t1: f64,
    p: f64,
    c2: f64,
}

impl VoltageSource {
    /// Creates a voltage source from a list of string parameters.
    ///
    /// `params[0]` is `"DC"`, `"SIN"` or `"PULSE"`.
    pub fn new(name: String, a: usize, b: usize, params: Vec<String>) -> Self {
        let mut base = ElementBase::new(name, a, b);

        let kind = params.get(0).cloned().unwrap_or_default();
        base.value = 0.0;

        let mut me = Self {
            base,
            kind,
            params,
            a0: 0.0,
            a: 0.0,
            f: 0.0,
            ta: 0.0,
            alpha: 0.0,
            k_deg: 0.0,
            c: 0.0,
            k2: 0.0,
            a2: 0.0,
            ta2: 0.0,
            ts: 0.0,
            td: 0.0,
            t1: 0.0,
            p: 0.0,
            c2: 0.0,
        };

        me.set_param();

        // Default: one period if c2 is not provided in the netlist.
        if me.kind == "PULSE" && me.c2 == 0.0 {
            me.c2 = 1.0;
        }

        me
    }

    /// Parses parameters for SIN or PULSE sources.
    fn set_param(&mut self) {
        if self.kind == "SIN" {
            self.a0 = self.params[1].parse().unwrap();
            self.a = self.params[2].parse().unwrap();
            self.f = self.params[3].parse().unwrap();
            self.ta = self.params[4].parse().unwrap();
            self.alpha = self.params[5].parse().unwrap();
            self.k_deg = self.params[6].parse().unwrap();
            self.c = self.params[7].parse().unwrap();
        } else if self.kind == "PULSE" {
            self.k2 = self.params[1].parse().unwrap();
            self.a2 = self.params[2].parse().unwrap();
            self.ta2 = self.params[3].parse().unwrap();
            self.ts = self.params[4].parse().unwrap();
            self.td = self.params[5].parse().unwrap();
            self.t1 = self.params[6].parse().unwrap();
            self.p = self.params[7].parse().unwrap();
        }
    }

    /// Updates `base.value` according to the waveform at time `time`.
    fn set_value(&mut self, time: f64, dt: f64) {
        if self.kind == "DC" {
            self.base.value = self.params[1].parse().unwrap();
            return;
        }

        if self.kind == "SIN" {
            let tf = (self.c / self.f) + self.ta;
            let mut t = time;

            if time <= self.ta {
                t = self.ta;
            } else if time > tf {
                t = tf;
            }

            self.base.value = self.a0
                + self.a
                    * (-self.alpha * (t - self.ta)).exp()
                    * (2.0 * PI * self.f * (t - self.ta) + PI * self.k_deg / 180.0).sin();
            return;
        }

        if self.kind == "PULSE" {
            let c2_eff = if self.c2 == 0.0 { 1.0 } else { self.c2 };
            let tf = (c2_eff * self.p) + self.ta2;
            let mut t = fmod_like(time - self.ta2, self.p);

            let mut ts = self.ts;
            let mut td = self.td;

            if ts == 0.0 {
                ts = dt;
            }
            if td == 0.0 {
                td = dt;
            }

            if time > self.ta2 && time <= tf {
                if t >= 0.0 && t < ts {
                    let m = (self.a2 - self.k2) / ts;
                    let b = self.k2;
                    self.base.value = m * t + b;
                } else if t >= ts && t <= (ts + self.t1) {
                    self.base.value = self.a2;
                } else if t > (ts + self.t1) && t <= (ts + self.t1 + td) {
                    let m = (self.k2 - self.a2) / td;
                    let b = -m * (ts + self.t1) + self.a2;
                    self.base.value = m * t + b;
                } else if t > (ts + self.t1 + td) {
                    self.base.value = self.k2;
                }
            } else if time <= self.ta2 || time > tf {
                self.base.value = self.k2;
            }
        }
    }
}

impl Element for VoltageSource {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the independent voltage source using branch current `jx`.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        time: f64,
        dt: f64,
        _sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let na = self.base.na;
        let nb = self.base.nb;
        let jx = self.base.jx;
        let s = system[0].len() - 1;

        self.set_value(time, dt);
        let v = self.base.value;

        system[na][jx] += 1.0;
        system[nb][jx] -= 1.0;
        system[jx][na] -= 1.0;
        system[jx][nb] += 1.0;
        system[jx][s] -= v;
    }
}

// ========================= Capacitor (C) =========================

/// Linear capacitor modeled with trapezoidal integration.
///
/// Uses an equivalent conductance and current source:
/// \(G = 2C / \Delta t\).
pub struct Capacitor {
    base: ElementBase,
    c: f64,
    v_prev: f64,
}

impl Capacitor {
    /// Creates a capacitor with capacitance `c_val` (farads).
    pub fn new(name: String, a: usize, b: usize, c_val: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.value = c_val;
        Self {
            base,
            c: c_val,
            v_prev: 0.0,
        }
    }
}

impl Element for Capacitor {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the trapezoidal equivalent for the capacitor.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        dt: f64,
        sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let na = self.base.na;
        let nb = self.base.nb;
        let c = self.c;
        let s = system[0].len() - 1;

        if dt <= 0.0 {
            return;
        }

        let g = 2.0 * c / dt;

        let v_prev_now = if na < sol_time.len() && nb < sol_time.len() {
            sol_time[na] - sol_time[nb]
        } else {
            0.0
        };

        let i_eq = g * v_prev_now;

        system[na][na] += g;
        system[nb][nb] += g;
        system[na][nb] -= g;
        system[nb][na] -= g;

        system[na][s] -= i_eq;
        system[nb][s] += i_eq;

        self.v_prev = v_prev_now;
    }
}

// ========================= Inductor (L) =========================

/// Linear inductor modeled with trapezoidal integration.
///
/// Branch current is represented by an auxiliary variable `jx`.
pub struct Inductor {
    base: ElementBase,
    l: f64,
    i_prev: f64,
}

impl Inductor {
    /// Creates an inductor with inductance `l_val` (henries).
    pub fn new(name: String, a: usize, b: usize, l_val: f64) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.value = l_val;
        Self {
            base,
            l: l_val,
            i_prev: 0.0,
        }
    }
}

impl Element for Inductor {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the trapezoidal equivalent for the inductor.
    ///
    /// Uses:
    /// \[
    /// i^{n+1} = i^{n} + \frac{\Delta t}{2L} (v^n + v^{n+1})
    /// \]
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        dt: f64,
        sol_time: &[f64],
        _sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        let na = self.base.na;
        let nb = self.base.nb;
        let jx = self.base.jx;
        let l = self.l;
        let s = system[0].len() - 1;

        if dt <= 0.0 || jx == 0 {
            return;
        }

        let v_prev = if na < sol_time.len() && nb < sol_time.len() {
            sol_time[na] - sol_time[nb]
        } else {
            0.0
        };

        // Branch connection for current jx
        system[na][jx] += 1.0;
        system[nb][jx] -= 1.0;
        system[jx][na] -= 1.0;
        system[jx][nb] += 1.0;

        let alpha = dt / (2.0 * l);

        system[jx][na] -= alpha;
        system[jx][nb] += alpha;

        let rhs = self.i_prev + alpha * v_prev;
        system[jx][s] -= rhs;

        self.i_prev = if jx < sol_time.len() {
            sol_time[jx]
        } else {
            0.0
        };
    }
}

// ========= Piecewise‑linear resistor (N) =========

/// Piecewise‑linear resistor defined by four (V, I) points.
///
/// Used as a generic nonlinear element in DC and transient analysis.
pub struct ResistorLinearPartes {
    base: ElementBase,

    // points (v1, j1, ... v4, j4)
    v1: f64,
    j1: f64,
    v2: f64,
    j2: f64,
    v3: f64,
    j3: f64,
    v4: f64,
    j4: f64,

    // Newton–Raphson node voltages
    vnr: Vec<f64>,

    // global gmin and convergence flags
    gm: f64,
    conv: Vec<bool>,

    // internal state for incremental stamping
    j0: f64,
    g0: f64,
    g: f64,
    j: f64,
    j0_prev: f64,
    g0_prev: f64,
    gm_prev: f64,
    k: f64,
    initial: bool,
}

impl ResistorLinearPartes {
    /// Creates a piecewise‑linear resistor from four (V, I) points.
    pub fn new(
        name: String,
        a: usize,
        b: usize,
        points: Vec<String>,
        vnr: Vec<f64>,
        gm: f64,
        conv: Vec<bool>,
    ) -> Self {
        let mut base = ElementBase::new(name, a, b);

        let v1 = points[0].parse().unwrap();
        let j1 = points[1].parse().unwrap();
        let v2 = points[2].parse().unwrap();
        let j2 = points[3].parse().unwrap();
        let v3 = points[4].parse().unwrap();
        let j3 = points[5].parse().unwrap();
        let v4 = points[6].parse().unwrap();
        let j4 = points[7].parse().unwrap();

        let n = if vnr.is_empty() { 2 } else { vnr.len() };

        base.value = 0.0;

        Self {
            base,
            v1,
            j1,
            v2,
            j2,
            v3,
            j3,
            v4,
            j4,
            vnr: vec![0.0; n],
            gm,
            conv: vec![true; n],
            j0: 0.0,
            g0: 0.0,
            g: 0.0,
            j: 0.0,
            j0_prev: 0.0,
            g0_prev: 0.0,
            gm_prev: 0.0,
            k: 0.0,
            initial: true,
        }
    }

    /// Computes the local linearization (g0, j0) for the current voltage.
    fn set_value(&mut self) {
        if self.vnr.is_empty() {
            self.g0 = 0.0;
            self.j0 = 0.0;
            return;
        }

        let v = if !self.initial {
            let na = self.base.na;
            let nb = self.base.nb;
            if na < self.vnr.len() && nb < self.vnr.len() {
                self.vnr[na] - self.vnr[nb]
            } else {
                0.1
            }
        } else {
            self.initial = false;
            0.1
        };

        if v < self.v2 {
            self.g0 = (self.j2 - self.j1) / (self.v2 - self.v1);
            self.j0 = self.j2 - self.g0 * self.v2;
        } else if v >= self.v2 && v < self.v3 {
            self.g0 = (self.j3 - self.j2) / (self.v3 - self.v2);
            self.j0 = self.j3 - self.g0 * self.v3;
        } else {
            self.g0 = (self.j4 - self.j3) / (self.v4 - self.v3);
            self.j0 = self.j4 - self.g0 * self.v4;
        }
    }
}

impl Element for ResistorLinearPartes {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the incremental conductance and current for Newton–Raphson.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        self.vnr = sol_nr.to_vec();
        self.set_value();

        self.g = self.g0 - self.g0_prev;
        self.j = self.j0 - self.j0_prev;

        let na = self.base.na;
        let nb = self.base.nb;
        let s = system[0].len() - 1;

        let mut k = 0.0;
        if na < self.conv.len() && nb < self.conv.len() {
            if !self.conv[na] || !self.conv[nb] {
                k = self.gm - self.gm_prev;
                self.gm_prev = self.gm;
            }
        }

        self.g0_prev = self.g0;
        self.j0_prev = self.j0;

        let geq = self.g + k;

        system[na][na] += geq;
        system[na][nb] -= geq;
        system[nb][na] -= geq;
        system[nb][nb] += geq;

        system[na][s] -= self.j;
        system[nb][s] += self.j;
    }
}

// ========= Nonlinear switch ($) =========

/// Voltage‑controlled nonlinear switch.
///
/// Conductance toggles between `gon` and `goff` depending on the
/// control voltage `V(nc) - V(nd)` compared to `vref`.
pub struct Chave {
    base: ElementBase,

    // electrical parameters
    gon: f64,
    goff: f64,
    vref: f64,

    // Newton–Raphson voltages
    vnr: Vec<f64>,

    // global gmin and convergence flags
    gm: f64,
    conv: Vec<bool>,

    // internal incremental state
    g: f64,
    g_prev: f64,
    gm_prev: f64,
    initial: bool,
}

impl Chave {
    /// Creates a new nonlinear switch `$`.
    pub fn new(
        name: String,
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        gon: f64,
        goff: f64,
        vref: f64,
        vnr: Vec<f64>,
        gm: f64,
        conv: Vec<bool>,
    ) -> Self {
        let mut base = ElementBase::new(name, a, b);
        base.nc = c;
        base.nd = d;
        base.value = 0.0;

        let n = if vnr.is_empty() { 2 } else { vnr.len() };

        Self {
            base,
            gon,
            goff,
            vref,
            vnr: vec![0.0; n],
            gm,
            conv: vec![true; n],
            g: 0.0,
            g_prev: 0.0,
            gm_prev: 0.0,
            initial: true,
        }
    }

    /// Computes the incremental conductance contribution.
    fn set_value(&mut self) {
        let nc = self.base.nc;
        let nd = self.base.nd;

        let vc = if nc < self.vnr.len() && nd < self.vnr.len() {
            self.vnr[nc] - self.vnr[nd]
        } else {
            0.0
        };

        let g_ideal = if vc >= self.vref { self.gon } else { self.goff };

        if self.initial {
            self.initial = false;
            self.g = g_ideal;
            self.g_prev = g_ideal;
        } else {
            self.g = g_ideal - self.g_prev;
            self.g_prev = g_ideal;
        }
    }
}

impl Element for Chave {
    fn base(&self) -> &ElementBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ElementBase {
        &mut self.base
    }

    /// Stamps the incremental conductance for the switch.
    fn add_stamp(
        &mut self,
        system: &mut Matrix,
        _time: f64,
        _dt: f64,
        _sol_time: &[f64],
        sol_nr: &[f64],
        _sol_prev_nr: &[f64],
        _converged: &[bool],
        _gmin: f64,
    ) {
        self.vnr = sol_nr.to_vec();

        self.set_value();

        let na = self.base.na;
        let nb = self.base.nb;
        let s = system[0].len() - 1;

        let mut k = 0.0;
        if na < self.conv.len() && nb < self.conv.len() {
            if !self.conv[na] || !self.conv[nb] {
                k = self.gm - self.gm_prev;
                self.gm_prev = self.gm;
            }
        }

        let geq = self.g + k;

        system[na][na] += geq;
        system[na][nb] -= geq;
        system[nb][na] -= geq;
        system[nb][nb] += geq;

        // ideal switch does not inject independent current
        system[na][s] -= 0.0;
        system[nb][s] += 0.0;
    }
}