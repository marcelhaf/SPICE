use crate::element::{
    Capacitor,
    Chave,
    CurrentControlledCurrentSource,  // F
    CurrentSource,                   // I
    Element,
    IdealTransformer,                // K
    Inductor,
    Matrix,
    OpAmp,                           // O
    Resistor,
    ResistorLinearPartes,            // N
    Transconductance,                // G
    Transresistance,                 // H
    VoltageControlledVoltageSource,  // E
    VoltageSource,                   // V
};

use std::fs::File;
use std::io::{BufRead, BufReader, Write};

/// Tolerance for detecting singular pivots in the linear solver.
const TOLG: f64 = 1e-12;

/// Circuit container: netlist, MNA matrix and analysis control.
pub struct Circuit {
    // analysis mode:
    // 1 - DC linear
    // 2 - DC nonlinear
    // 3 - Transient (SP) linear (SIN / PULSE)
    // 4 - Transient (SP) nonlinear
    mode: i32,
    sp: bool,
    linear: bool,

    num_vars: usize,
    num_elements: usize,
    num_nodes: usize,

    /// Effective time step used when writing the table (may be
    /// multiplied by `table_stride`).
    dt: f64,
    /// Total simulation time for transient analysis.
    total_time: f64,
    /// Current simulation time.
    time: f64,

    /// Stride used when sampling output for the table.
    table_stride: usize,
    file_name: String,

    /// Global gmin used for Newton–Raphson stepping (also used by N and `$`).
    gmin: f64,

    // Netlists:
    // netlist1: elements with time‑independent stamps (R, controlled sources, etc.)
    // netlist2: time‑dependent elements (V, I, C, L, etc.)
    // netlist3: nonlinear elements (N, switch)
    netlist1: Vec<Box<dyn Element>>,
    netlist2: Vec<Box<dyn Element>>,
    netlist3: Vec<Box<dyn Element>>,

    /// Current MNA system matrix.
    system: Matrix,
    /// Last solved MNA system (after Gaussian elimination).
    solved_system: Matrix,

    /// Result table: each row is a time point, columns are variables.
    table: Vec<Vec<f64>>,
    /// Variable names: node voltages and branch currents (`jV1`, etc.).
    variables: Vec<String>,

    // Vectors used by Newton–Raphson
    /// Convergence flags for each variable.
    converged: Vec<bool>,
    /// Solution at the current time (linear or last NR iteration).
    sol_time: Vec<f64>,
    /// Previous Newton–Raphson solution.
    sol_prev_nr: Vec<f64>,
    /// Current Newton–Raphson solution.
    sol_nr: Vec<f64>,
}

impl Circuit {
    /// Creates an empty circuit with default settings.
    pub fn new() -> Self {
        Self {
            mode: 1,
            sp: false,
            linear: true,

            num_vars: 0,
            num_elements: 0,
            num_nodes: 0,

            dt: 0.0,
            total_time: 0.0,
            time: 0.0,

            table_stride: 0,
            file_name: String::new(),

            gmin: 0.0,

            netlist1: vec![],
            netlist2: vec![],
            netlist3: vec![],

            system: vec![],
            solved_system: vec![],

            table: vec![],
            variables: vec!["0".to_string()], // ground node

            converged: vec![],
            sol_time: vec![],
            sol_prev_nr: vec![],
            sol_nr: vec![],
        }
    }

    /// Runs the full analysis for a given netlist file.
    ///
    /// # Arguments
    /// * `file_name` - Path to a `.net` file.
    ///
    /// # Errors
    /// Returns an error string if the netlist is invalid,
    /// the system is singular, or Newton–Raphson does not converge.
    pub fn run_system(&mut self, file_name: &str) -> Result<(), String> {
        self.file_name = file_name.to_string();
        let lines = self.read_file(file_name)?;
        self.build_netlist(lines)?;

        println!(
            "mode = {}, dt = {}, total_time = {}",
            self.mode, self.dt, self.total_time
        );

        match self.mode {
            1 => self.run_dc_linear()?,
            2 => self.run_dc_nonlinear()?,
            3 => self.run_sp_linear()?,
            4 => self.run_sp_nonlinear()?,
            _ => return Err("Invalid analysis mode".to_string()),
        }

        self.save_table()?;
        Ok(())
    }

    /// Runs a linear DC operating point analysis (no nonlinear elements).
    fn run_dc_linear(&mut self) -> Result<(), String> {
        self.time = 0.0;
        self.build_system_base();
        self.apply_time_elements();
        if self.solve_system() {
            return Err("Singular system".to_string());
        }
        self.save_solution_time();
        Ok(())
    }

    /// Runs a nonlinear DC operating point analysis using Newton–Raphson
    /// and gmin stepping.
    fn run_dc_nonlinear(&mut self) -> Result<(), String> {
        const MAX_ERROR: f64 = 0.01;
        const MAX_IT: i32 = 5;

        let mut error: f64;
        let mut gmin_prev: f64 = 500.0;
        let mut iter_count: i32 = 1;
        let mut step_index: usize = 0;
        let mut final_solution = true;
        let mut g_stepping = false;
        let mut last_good_solution: Vec<f64> = Vec::new();

        let factor: [f64; 5] = [
            0.0,
            10.0_f64.sqrt(),
            10.0_f64.sqrt().sqrt(),
            10.0_f64.sqrt().sqrt().sqrt(),
            10.0_f64.sqrt().sqrt().sqrt().sqrt(),
        ];

        // Initial solution with gmin = 0
        self.build_system_base();
        self.apply_time_elements();
        self.apply_nonlinear_elements();
        if self.solve_system() {
            return Err("Singular system".to_string());
        }
        self.save_solution_nr();
        error = self.get_error(false);

        while error > MAX_ERROR || !final_solution {
            if iter_count > MAX_IT {
                g_stepping = true;
                final_solution = false;
                iter_count = 0;

                if step_index > 4 {
                    return Err("System does not converge".to_string());
                }

                if step_index == 0 {
                    self.gmin = 1.1;
                } else {
                    self.gmin = gmin_prev / factor[step_index];
                    self.sol_nr = last_good_solution.clone();
                }
                step_index += 1;
            }

            self.get_error(g_stepping);
            self.sol_prev_nr = self.sol_nr.clone();

            self.modify_system_nr();
            if self.solve_system() {
                return Err("Singular system".to_string());
            }
            self.save_solution_nr();
            error = self.get_error(g_stepping);
            iter_count += 1;

            if g_stepping && error <= MAX_ERROR {
                if self.gmin > 1e-12 {
                    last_good_solution = self.sol_nr.clone();
                    gmin_prev = self.gmin;
                    self.gmin /= 10.0;
                    iter_count = 0;
                    step_index = 1;
                } else {
                    final_solution = true;
                }
            }
        }

        self.table.push(self.sol_nr.clone());
        Ok(())
    }

    /// Computes the maximum Newton–Raphson error between
    /// `sol_nr` and `sol_prev_nr`.
    ///
    /// If `mark_not_converged` is true, marks variables with
    /// error > 1e-6 as not converged in `converged`.
    fn get_error(&mut self, mark_not_converged: bool) -> f64 {
        let mut max = -1.0;

        for i in 1..self.variables.len() {
            let diff = self.sol_nr[i] - self.sol_prev_nr[i];
            let err = if self.sol_nr[i].abs() > 1.0 {
                (diff / self.sol_nr[i]).abs()
            } else {
                diff.abs()
            };

            if err > 1e-6 && mark_not_converged {
                self.converged[i] = false;
            }

            if err > max {
                max = err;
            }
        }
        max
    }

    /// Runs a linear transient (SP) analysis.
    fn run_sp_linear(&mut self) -> Result<(), String> {
        self.time = 0.0;

        self.build_system_base();
        self.apply_time_elements();
        if self.solve_system() {
            return Err(format!("Singular system at t = {}", self.time));
        }
        self.save_solution_time();

        let dt = self.dt;
        let mut t = dt;
        while t <= self.total_time + 1e-12 {
            self.time = t;

            self.build_system_base();
            self.apply_time_elements();

            if self.solve_system() {
                return Err(format!("Singular system at t = {}", self.time));
            }
            self.save_solution_time();
            t += dt;
        }

        Ok(())
    }

    /// Runs a nonlinear transient (SP) analysis using Newton–Raphson
    /// and gmin stepping at each time point.
    fn run_sp_nonlinear(&mut self) -> Result<(), String> {
        const MAX_ERROR: f64 = 1e-6;
        const MAX_IT: i32 = 6;

        let mut _gmin_steps_count: i32 = 0; // kept for debugging parity with C++

        let factor: [f64; 5] = [
            0.0,
            10.0_f64.sqrt(),
            10.0_f64.sqrt().sqrt(),
            10.0_f64.sqrt().sqrt().sqrt(),
            10.0_f64.sqrt().sqrt().sqrt().sqrt(),
        ];

        let dt = self.dt;
        let mut t = 0.0;
        self.time = 0.0;

        while t <= self.total_time + 1e-12 {
            self.time = t;

            // Reset Newton control for this time step
            let mut error: f64 = 0.0;
            let mut gmin_prev: f64 = 1.1;
            let mut iter_count: i32 = 1;
            let mut step_index: usize = 0;
            let mut final_solution = true;
            let mut g_stepping = false;
            let mut last_good_solution: Vec<f64> = Vec::new();

            // Reset gmin and convergence flags
            self.gmin = 0.0;
            self.converged.fill(true);

            // Initial solution at this time
            self.build_system_base();
            self.apply_time_elements();

            self.apply_nonlinear_elements();
            if self.solve_system() {
                return Err(format!("Singular system at t = {}", self.time));
            }
            self.save_solution_nr();
            error = self.get_error(false);

            // Newton–Raphson loop
            while error > MAX_ERROR || !final_solution {
                if iter_count > MAX_IT {
                    g_stepping = true;
                    final_solution = false;
                    iter_count = 0;

                    if step_index > 4 {
                        return Err("System does not converge.".to_string());
                    }

                    if step_index == 0 {
                        self.gmin = 1.1;
                    } else {
                        self.gmin = gmin_prev / factor[step_index];
                        self.sol_nr = last_good_solution.clone();
                    }
                    step_index += 1;
                    _gmin_steps_count += 1;
                }

                self.get_error(g_stepping);
                self.sol_prev_nr = self.sol_nr.clone();

                self.modify_system_nr();
                if self.solve_system() {
                    return Err(format!("Singular system at t = {}", self.time));
                }
                self.save_solution_nr();
                error = self.get_error(g_stepping);
                iter_count += 1;

                if g_stepping && error <= MAX_ERROR {
                    if self.gmin > 1e-12 {
                        last_good_solution = self.sol_nr.clone();
                        gmin_prev = self.gmin;
                        self.gmin /= 10.0;
                        iter_count = 0;
                        step_index = 1;
                    } else {
                        final_solution = true;
                    }
                }
            }

            // Converged at this time: store solution in table
            let mut row = vec![0.0];
            for i in 1..=self.num_vars {
                row.push(self.sol_nr[i]);
            }
            self.table.push(row);
            t += dt;
        }
        println!("SPN: table.len() = {}", self.table.len());
        Ok(())
    }

    /// Reads all lines from a netlist file into a vector of strings.
    fn read_file(&self, file_name: &str) -> Result<Vec<String>, String> {
        let f = File::open(file_name).map_err(|_| "Invalid file".to_string())?;
        let reader = BufReader::new(f);
        let mut resp = vec![];
        for line in reader.lines() {
            resp.push(line.map_err(|_| "Invalid file".to_string())?);
        }
        Ok(resp)
    }

    /// Builds internal netlists and analysis configuration from
    /// the netlist text lines.
    fn build_netlist(&mut self, file_lines: Vec<String>) -> Result<(), String> {
        let mut tran = false;

        // start at 1 because line 0 is the number of nodes in the original C++ format
        for i in 1..file_lines.len() {
            let mut tokens = Self::split(&file_lines[i]);
            if tokens.is_empty() {
                continue;
            }

            // .TRAN command
            if tokens[0] == ".TRAN" {
                tran = true;
                // tokens[1] = tstart (ignored)
                self.total_time = tokens
                    .get(2)
                    .ok_or("Invalid command")?
                    .parse()
                    .map_err(|_| "Invalid command".to_string())?;

                self.dt = tokens
                    .get(3)
                    .ok_or("Invalid command")?
                    .parse()
                    .map_err(|_| "Invalid command".to_string())?;

                let stride_f: f64 = tokens
                    .get(4)
                    .ok_or("Invalid command")?
                    .parse()
                    .map_err(|_| "Invalid command".to_string())?;
                self.table_stride = stride_f as usize;
                // never divide: C++ multiplies dt by stride
                self.dt = self.dt * (self.table_stride as f64);

                continue;
            }

            // element type from first character
            let first_char = tokens[0]
                .chars()
                .next()
                .unwrap_or('*')
                .to_ascii_uppercase();
            if first_char == '*' {
                continue; // comment
            }

            // force first letter of element name to upper case
            if !tokens[0].is_empty() {
                let mut chars: Vec<char> = tokens[0].chars().collect();
                chars[0] = chars[0].to_ascii_uppercase();
                tokens[0] = chars.into_iter().collect();
            }

            self.add_element(first_char, &tokens)?;
        }

        if !tran {
            return Err("Invalid command".to_string());
        }

        self.num_nodes = self.variables.len() - 1;
        self.add_current_variables();

        self.num_elements = self.netlist1.len() + self.netlist2.len() + self.netlist3.len();
        self.num_vars = self.variables.len() - 1;

        // Allocate solution vectors
        self.sol_nr.clear();
        self.sol_nr.resize(self.variables.len(), 0.0);

        self.sol_prev_nr.clear();
        self.sol_prev_nr.resize(self.variables.len(), 0.0);

        self.sol_time.clear();
        self.sol_time.resize(self.variables.len(), 0.0);

        self.converged.clear();
        self.converged.resize(self.variables.len(), true);

        // Choose analysis mode (same logic as C++):
        // sp = true  -> time analysis (SIN/PULSE present)
        // linear = false -> nonlinear elements (N or $) present
        self.mode = if self.sp && self.linear {
            3 // SP linear
        } else if !self.sp && !self.linear {
            2 // DC nonlinear
        } else if self.sp && !self.linear {
            4 // SP nonlinear
        } else {
            1 // DC linear
        };

        Ok(())
    }

    /// Adds one element to the appropriate internal netlist.
    fn add_element(&mut self, kind: char, data: &[String]) -> Result<(), String> {
        let a = self.node_index(&data[1]);
        let b = self.node_index(&data[2]);

        match kind {
            'R' => {
                let r: f64 = data[3]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1
                    .push(Box::new(Resistor::new(data[0].clone(), a, b, r)));
            }
            'V' => {
                let params = data[3..].to_vec();
                if params.get(0).map(|s| s.as_str()) == Some("SIN")
                    || params.get(0).map(|s| s.as_str()) == Some("PULSE")
                {
                    self.sp = true;
                }
                self.netlist2
                    .push(Box::new(VoltageSource::new(data[0].clone(), a, b, params)));
            }
            'I' => {
                let params = data[3..].to_vec();
                if params.get(0).map(|s| s.as_str()) == Some("SIN")
                    || params.get(0).map(|s| s.as_str()) == Some("PULSE")
                {
                    self.sp = true;
                }
                self.netlist2
                    .push(Box::new(CurrentSource::new(data[0].clone(), a, b, params)));
            }
            'C' => {
                self.sp = true;
                let c_val: f64 = data[3]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist2
                    .push(Box::new(Capacitor::new(data[0].clone(), a, b, c_val)));
            }
            'L' => {
                self.sp = true;
                let l_val: f64 = data[3]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist2
                    .push(Box::new(Inductor::new(data[0].clone(), a, b, l_val)));
            }
            'G' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                let g_val: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1.push(Box::new(Transconductance::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    g_val,
                )));
            }
            'E' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                let gain: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1.push(Box::new(VoltageControlledVoltageSource::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    gain,
                )));
            }
            'F' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                let gain: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1.push(Box::new(CurrentControlledCurrentSource::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    gain,
                )));
            }
            'H' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                let r_val: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1.push(Box::new(Transresistance::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    r_val,
                )));
            }
            'O' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                self.netlist1
                    .push(Box::new(OpAmp::new(data[0].clone(), a, b, c, d)));
            }
            'K' => {
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);
                let n: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                self.netlist1.push(Box::new(IdealTransformer::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    n,
                )));
            }
            'N' => {
                self.linear = false;
                let points = data[3..].to_vec();
                self.netlist3.push(Box::new(ResistorLinearPartes::new(
                    data[0].clone(),
                    a,
                    b,
                    points,
                    self.sol_nr.clone(),
                    self.gmin,
                    self.converged.clone(),
                )));
            }
            '$' => {
                // $ <name> na nb nc nd gon goff vref
                self.linear = false;
                let c = self.node_index(&data[3]);
                let d = self.node_index(&data[4]);

                let gon: f64 = data[5]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                let goff: f64 = data[6]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;
                let vref: f64 = data[7]
                    .parse()
                    .map_err(|_| "Unknown element".to_string())?;

                self.netlist3.push(Box::new(Chave::new(
                    data[0].clone(),
                    a,
                    b,
                    c,
                    d,
                    gon,
                    goff,
                    vref,
                    self.sol_nr.clone(),
                    self.gmin,
                    self.converged.clone(),
                )));
            }
            _ => return Err("Unknown element".to_string()),
        }
        Ok(())
    }

    /// Adds auxiliary current variables for controlled sources and
    /// elements that require branch currents (V, L, E, F, H, O, K).
    fn add_current_variables(&mut self) {
        // netlist1: E, F, O, H, K
        for e in self.netlist1.iter_mut() {
            let name = e.base().name.clone();
            let kind = name.chars().next().unwrap_or(' ');

            if kind == 'E' || kind == 'F' || kind == 'O' || kind == 'K' {
                self.variables.push(format!("j{}", name));
                let idx = self.variables.len() - 1;
                e.base_mut().jx = idx;
            } else if kind == 'H' {
                self.variables.push(format!("jx{}", name));
                self.variables.push(format!("jy{}", name));
                let jx_idx = self.variables.len() - 2;
                let jy_idx = self.variables.len() - 1;
                e.base_mut().jx = jx_idx;
                e.base_mut().jy = jy_idx;
            }
        }

        // netlist2: V and L get a branch current jx
        for e in self.netlist2.iter_mut() {
            let name = e.base().name.clone();
            let kind = name.chars().next().unwrap_or(' ');

            if kind == 'V' || kind == 'L' {
                self.variables.push(format!("j{}", name));
                let idx = self.variables.len() - 1;
                e.base_mut().jx = idx;
            }
        }
    }

    /// Returns the index of a node name in `variables`, inserting it if needed.
    fn node_index(&mut self, name: &str) -> usize {
        if let Some(pos) = self.variables.iter().position(|s| s == name) {
            pos
        } else {
            self.variables.push(name.to_string());
            self.variables.len() - 1
        }
    }

    /// Builds an empty MNA system matrix and stamps all time‑independent elements.
    fn build_system_base(&mut self) {
        let rows = self.num_vars + 1;
        let cols = self.num_vars + 2;
        self.system = vec![vec![0.0; cols]; rows];

        let time = self.time;
        let dt = self.dt;

        for e in self.netlist1.iter_mut() {
            e.add_stamp(
                &mut self.system,
                time,
                dt,
                &self.sol_time,
                &self.sol_nr,
                &self.sol_prev_nr,
                &self.converged,
                self.gmin,
            );
        }
    }

    /// Stamps all time‑dependent elements (V, I, C, L, etc.) onto the system.
    fn apply_time_elements(&mut self) {
        let time = self.time;
        let dt = self.dt;

        for e in self.netlist2.iter_mut() {
            e.add_stamp(
                &mut self.system,
                time,
                dt,
                &self.sol_time,
                &self.sol_nr,
                &self.sol_prev_nr,
                &self.converged,
                self.gmin,
            );
        }
    }

    /// Solves the current linear system using Gaussian elimination
    /// with partial pivoting.
    ///
    /// # Returns
    /// `true` if the matrix is singular, `false` otherwise.
    fn solve_system(&mut self) -> bool {
        let mut vec = self.system.clone();

        for i in 1..=self.num_vars {
            let mut t: f64 = 0.0;
            let mut a = i;

            // pivot search
            for l in i..=self.num_vars {
                if vec[l][i].abs() > t.abs() {
                    a = l;
                    t = vec[l][i];
                }
            }

            // row swap
            if i != a {
                for l in 1..=self.num_vars + 1 {
                    let p = vec[i][l];
                    vec[i][l] = vec[a][l];
                    vec[a][l] = p;
                }
            }

            if t.abs() < TOLG {
                return true;
            }

            // normalize pivot row and eliminate column
            for j in (1..=self.num_vars + 1).rev() {
                vec[i][j] /= t;
                let p = vec[i][j];
                if p != 0.0 {
                    for l in 1..=self.num_vars {
                        if l != i {
                            vec[l][j] -= vec[l][i] * p;
                        }
                    }
                }
            }
        }

        self.solved_system = vec;
        false
    }

    /// Stores the current linear solution into `table` and `sol_time`.
    fn save_solution_time(&mut self) {
        let mut aux = vec![0.0];
        for i in 1..=self.num_vars {
            aux.push(self.solved_system[i][self.num_vars + 1]);
        }
        self.table.push(aux.clone());
        self.sol_time = aux;
    }

    /// Writes the result table to a `.tab` file.
    fn save_table(&self) -> Result<(), String> {
        let out_name = if let Some(pos) = self.file_name.find(".net") {
            let mut s = self.file_name.clone();
            s.replace_range(pos..(pos + 4), ".tab");
            s
        } else {
            format!("{}.tab", self.file_name)
        };

        let mut f = File::create(out_name).map_err(|_| "Failed to write output".to_string())?;

        // header
        write!(f, "t").map_err(|_| "Failed to write output".to_string())?;
        for k in 1..self.variables.len() {
            write!(f, " {}", self.variables[k])
                .map_err(|_| "Failed to write output".to_string())?;
        }
        writeln!(f).map_err(|_| "Failed to write output".to_string())?;

        if self.mode == 1 {
            // DC linear: single row from table[0]
            write!(f, "0").map_err(|_| "Failed to write output".to_string())?;
            if !self.table.is_empty() {
                for j in 1..self.table[0].len() {
                    write!(f, " {}", self.table[0][j])
                        .map_err(|_| "Failed to write output".to_string())?;
                }
            }
            writeln!(f).map_err(|_| "Failed to write output".to_string())?;
        } else if self.mode == 2 {
            // DC nonlinear: use final Newton solution sol_nr
            write!(f, "0").map_err(|_| "Failed to write output".to_string())?;
            if !self.sol_nr.is_empty() {
                for i in 1..self.sol_nr.len() {
                    write!(f, " {}", self.sol_nr[i])
                        .map_err(|_| "Failed to write output".to_string())?;
                }
            }
            writeln!(f).map_err(|_| "Failed to write output".to_string())?;
        } else if self.mode == 3 || self.mode == 4 {
            println!("save_table: mode = {}, rows = {}", self.mode, self.table.len());
            // transient SP: iterate over table rows
            let dt = self.dt;
            for (idx, row) in self.table.iter().enumerate() {
                let t = idx as f64 * dt;
                write!(f, "{}", t).map_err(|_| "Failed to write output".to_string())?;
                for j in 1..row.len() {
                    write!(f, " {}", row[j])
                        .map_err(|_| "Failed to write output".to_string())?;
                }
                writeln!(f).map_err(|_| "Failed to write output".to_string())?;
            }
        }

        Ok(())
    }

    /// Splits a line by whitespace into tokens.
    fn split(s: &str) -> Vec<String> {
        s.split_whitespace().map(|x| x.to_string()).collect()
    }

    /// Stamps all nonlinear elements (N and `$`) onto the system.
    fn apply_nonlinear_elements(&mut self) {
        for e in self.netlist3.iter_mut() {
            e.add_stamp(
                &mut self.system,
                self.time,
                self.dt,
                &self.sol_time,
                &self.sol_nr,
                &self.sol_prev_nr,
                &self.converged,
                self.gmin,
            );
        }
    }

    /// Rebuilds the nonlinear part of the system for the next
    /// Newton–Raphson iteration.
    fn modify_system_nr(&mut self) {
        // Nonlinear elements read `sol_nr` via the parameters passed to
        // `add_stamp`, so there is no need to update an internal field here.
        self.apply_nonlinear_elements();
    }

    /// Stores the current Newton–Raphson solution from `solved_system` into `sol_nr`.
    fn save_solution_nr(&mut self) {
        let mut aux = vec![0.0];
        for i in 1..=self.num_vars {
            aux.push(self.solved_system[i][self.num_vars + 1]);
        }
        self.sol_nr = aux;
    }
}