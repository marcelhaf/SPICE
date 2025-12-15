mod circuit;
mod element;

use std::io::{self, Write};

/// Entry point of the simulator.
///
/// The program:
/// 1. Asks the user for a netlist file name.
/// 2. Builds and runs the corresponding circuit analysis.
/// 3. Writes the results to a `.tab` file next to the netlist.
fn main() {
    let mut circuit = circuit::Circuit::new();

    print!("Enter netlist file name:\n");
    let _ = io::stdout().flush();

    let mut file_name = String::new();
    if io::stdin().read_line(&mut file_name).is_err() {
        eprintln!("Failed to read file name");
        return;
    }
    let file_name = file_name.trim().to_string();

    match circuit.run_system(&file_name) {
        Ok(()) => println!("Simulation finished successfully."),
        Err(msg) => println!("{msg}"),
    }

    // Simple pause for console use (ignored if input fails).
    let mut _pause = String::new();
    let _ = io::stdin().read_line(&mut _pause);
}