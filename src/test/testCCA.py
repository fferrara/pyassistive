from util import config, preprocessing, processing
from util.processing import Processing

FREQUENCIES = [6.4, 8.] # SX, DX
cca = processing.CCA(list(FREQUENCIES), 512)
ref = Processing.generate_references(512, 6.4)

# --> [a b r] = canoncorr(vetor(:,1),vetor(:,2))
print cca._compute_max_corr(ref[:, 0:1], ref[:, 1:2])
# --> [a b r] = canoncorr(vetor(:,2),vetor(:,3))
print cca._compute_max_corr(ref[:, 1:2], ref[:, 2:3])