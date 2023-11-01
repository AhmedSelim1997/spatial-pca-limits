# Extract command-line arguments
N="$1"
SIGMA="$2"
T_END="$3"
OUTPUT_DIR="$4"

# Create the SLiM script
cat <<EOF > ${OUTPUT_DIR}/simulation.slim
initialize() {
	defineConstant("N",$N);
    defineConstant("sigma",$SIGMA);
	initializeSLiMOptions(dimensionality="xy");
	initializeMutationRate(1e-5);
	initializeMutationType("m1", 0.5, "f", 0.0);
	initializeGenomicElementType("g1", m1, 1.0);
	initializeGenomicElement(g1, 0, 99999);
	initializeRecombinationRate(1e-3);
	// spatial competition
	initializeInteractionType(1, "xy", reciprocal=T, maxDistance=0.3);
	i1.setInteractionFunction("n", 3.0, 0.1);
	// spatial mate choice
	initializeInteractionType(2, "xy", reciprocal=T, maxDistance=0.1);
}
1 late() {
	sim.addSubpop("p1", N);
	p1.individuals.setSpatialPosition(p1.pointUniform(N)); // initialize the individuals uniformly over space
}
1: late() {
	i1.evaluate(p1);
	inds = sim.subpopulations.individuals;
	competition = i1.totalOfNeighborStrengths(inds); // Computing competition strengths of each ind with all others
	inds.fitnessScaling = 1.1 - competition / size(inds); // computing fitness of each ind
}
2: first() {
	i2.evaluate(p1);
}
mateChoice() {
	// nearest-neighbor mate choice
	neighbors = i2.nearestNeighbors(individual, 3);
	return (size(neighbors) ? sample(neighbors, 1) else float(0));
}

modifyChild() {
	// Reflecting boundary conditions
	pos = parent1.spatialPosition + rnorm(2, 0, sigma);
	child.setSpatialPosition(p1.pointReflected(pos));
	return T;
}

$T_END late() { 
	sim.outputFull(filePath="full.txt",ages=F,ancestralNucleotides=F); 
	}
EOF