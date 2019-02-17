#coding:utf-8
from tree.regression_tree import RegressionTree
from utils.load_data import load_boston_house_prices
from utils.model_selection import get_r2, train_test_split
from utils.utils import run_time 

@run_time
def main():
    print("Testing the performance of RegressionTree...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # Train model
    reg = RegressionTree()
    reg.fit(X=x_train, y=y_train, max_depth=5)
    # Show rules
    reg.rules
    # Model evaluation
    get_r2(reg, x_test, y_test)

if __name__ == "__main__":
    main()
