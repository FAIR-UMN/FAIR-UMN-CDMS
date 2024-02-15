import numpy as np


dataset = np.loadtxt('/home/submit/aidandc/SuperCDMS/trainingDataFloatHOS.csv',delimiter=',')
test_dataset = np.loadtxt('/home/submit/aidandc/SuperCDMS/testingDataFloatHOS.csv',delimiter=',')

X1 = dataset[:, 1]
X2 = dataset[:, 4]
X3 = dataset[:, 3]

X = []
for i in range(len(X1)):
    X.append((X1[i],X2[i],X3[i]))
X = np.array(X)

#X = dataset[:,1]
#X = np.reshape(X, (len(X),1))
y = dataset[:,-1]

X_test1 = test_dataset[:,1]
X_test2 = test_dataset[:,4]
X_test3 = test_dataset[:,3]

X_test = []
for i in range(len(X_test1)):
    X_test.append((X_test1[i],X_test2[i],X_test3[i]))
X_test = np.array(X_test)

#X_test = np.reshape(X_test, (len(X_test),1))
y_test = test_dataset[:,-1]

from pysr import PySRRegressor

model = PySRRegressor(
    model_selection="best",  # Result is mix of simplicity+accuracy
    niterations=40,
    binary_operators=["+", "*"],
    #timeout_in_seconds=5,
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
	# ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss="loss(x, y) = (x - y)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, y)

print(model)

y_preds = model.predict(X_test)

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

print(rmse(y_preds,y_test))