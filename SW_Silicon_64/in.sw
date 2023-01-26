# bulk Si via Stillinger-Weber

units		metal
atom_style	atomic
newton      on
lattice		diamond 5.431
region		box block 0 2 0 2 0 2
create_box	1 box
create_atoms	1 box
timestep	0.001

pair_style	sw
pair_coeff	* * Si.sw Si
mass            1 28.06

velocity	all create 500.0 376847 loop geom
dump            2 all custom 100 md_min.lammpstrj id type mass x y z vx vy vz fx fy fz
minimize	    1.0e-10 1.0e-10 10000 100000
undump          2


fix             1 all npt temp 3500 3500 0.3 iso 0.0 0.0 1000		
dump            2 all custom 1000 md_npt.lammpstrj id type mass x y z vx vy vz fx fy fz
run             100000
unfix           1
undump          2

neighbor	1.0 bin
neigh_modify    delay 5 every 1


reset_timestep 0
fix		1 all nve
thermo      100
dump            2 all custom 1 md_nve.lammpstrj id type mass x y z vx vy vz fx fy fz
run		10000
unfix   1
undump   2
