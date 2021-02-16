from model_with_uncertainty.molecule_models import MolMCDropGNN


def load_dataset():
    x = np.random.uniform(size=(10000,1024))
    func = np.random.uniform(size=(1024,1)) / 1024.
    y = np.matmul(x, func)
    return x,y


def standardize_dataset(x, y):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    return x, y


def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_test, y_test = torch.tensor(x_test).float(), torch.tensor(y_test).float()
    return x_train, y_train, x_test, y_test


x, y = load_dataset()
print("loaded dataset")
x, y = standardize_dataset(x, y)
print("standardized dataset")
x_train, y_train, x_test, y_test = split_dataset(x, y)
print("splitted dataset")

model = MolMCDropGNN()
model.fit()