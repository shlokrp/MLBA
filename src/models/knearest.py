from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def build_knn_classifier(n=5):
    return KNeighborsClassifier(n_neighbors=n)

def build_knn_regressor(n=5):
    return KNeighborsRegressor(n_neighbors=n)
