from sklearn.cluster import SpectralClustering
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score as nmi

from rsc.clustering import RSC

X, y = datasets.make_moons(600, shuffle=False, random_state=4, noise=0.1)
k = y.max()+1
nn = 15

rsc = RSC(k=k, nn=nn, theta=10)
y_rsc = rsc.fit_predict(X)

sc = SpectralClustering(n_clusters=k, n_neighbors=nn+1, affinity='nearest_neighbors')  # nn+1 since they include self
y_sc = sc.fit_predict(X)

print('SC NMI: {:.4f}, RSC NMI: {:.4f}'.format(nmi(y, y_sc), nmi(y, y_rsc)))