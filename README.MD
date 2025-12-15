# SPICE

A small SPICE-like circuit simulator written in Rust.
It supports DC and transient analyses, linear and nonlinear elements, and reads a simple netlist format.

## Features

- DC operating point analysis  
  - Linear (mode 1)  
  - Nonlinear with Newton–Raphson and gmin stepping (mode 2)
- Transient (time-domain) analysis  
  - Linear circuits with SIN / PULSE sources, capacitors and inductors (mode 3)  
  - Nonlinear circuits with piecewise-linear resistors (N) and switches (`$`) (mode 4)
- Elements (SPICE-like notation)
  - `R` – Resistor
  - `C` – Capacitor
  - `L` – Inductor
  - `V` – Independent voltage source (DC, SIN, PULSE)
  - `I` – Independent current source (DC, SIN, PULSE)
  - `G` – VCCS (voltage controlled current source)
  - `E` – VCVS (voltage controlled voltage source)
  - `F` – CCCS (current controlled current source)
  - `H` – CCVS (current controlled voltage source)
  - `O` – Ideal operational amplifier
  - `K` – Ideal transformer
  - `N` – Piecewise-linear resistor
  - `$` – Nonlinear voltage-controlled switch

## Netlist format

Netlists are plain text files. The first line is usually ignored for node count
(kept for compatibility with the original C++ project). Each subsequent line
defines an element or a control statement.

Example transient netlist:

``` example.net
3
V1 1 0 PULSE 0 10 0 1e-6 1e-6 5e-3 1e-2
$S1 1 2 1 0 1e3 1e-3 5
R1 2 0 1000
Rleak 1 0 1e9
.TRAN 0 0.02 1e-4 1
```

- `.TRAN tstart tstop dt stride`
  - `tstart` is currently ignored (simulation starts at 0).
  - `tstop` is the final time.
  - `dt` is the basic time step.
  - `stride` is a sampling factor; the effective step is `dt * stride`.

The simulator writes results to a `.tab` file with the same base name:
``` example.tab
t 1 2 jV1
0 0 0 0
0.0001 10 ...
...
```

Columns correspond to node voltages and branch currents (`j<element>`).

## Building

This is a standard Cargo binary crate.

```
cargo build
```

## Running

```
cargo run
```


The program will ask for a netlist file name:
```
Enter netlist file name:
>>> TRANSIENT.net
```

After the simulation finishes, a `TRANSIENT.tab` file is created in the same directory.


# License
This project is under MIT License.