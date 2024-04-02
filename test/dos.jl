@testitem "density of states (dos, ldos) computation for silicon" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    testcase = TestCases.silicon

    atoms=fill(ElementPsp(testcase.atnum; psp=load_psp(testcase.psp_upf)), 2)
	model = model_LDA(testcase.lattice, atoms, testcase.positions; temperature=1e-2)
	basis = PlaneWaveBasis(model; Ecut=12, testcase.kgrid)
	scfres = self_consistent_field(basis, tol=1e-8);

	plot_dos(scfres)
	plot_ldos(scfres)
end

@testitem "projected density of states computation for silicon" #=
    =#    tags=[:dont_test_mpi] setup=[TestCases] begin
    using DFTK
    testcase = TestCases.silicon

    atoms=fill(ElementPsp(testcase.atnum; psp=load_psp(testcase.psp_upf)), 2)
	for a in collect(1:0.2:2)
		model = model_LDA(a * testcase.lattice, atoms, 1.5.*testcase.positions; temperature=5e-3)
		basis = PlaneWaveBasis(model; Ecut=12, testcase.kgrid)
		scfres = self_consistent_field(basis, tol=1e-8);
		plot_pdos(1,scfres)
	end
end