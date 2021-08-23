import unittest
from test.debuggable_test_case import DebuggableTestCase
from depot import VerlustTopf
from depot import Depot
import numpy as np

class TestDepot(DebuggableTestCase):
    def setUp(self):
        self.taxRate = 0.26
        teilfreistellung = 0.3
        self.verlustTopf = VerlustTopf()
        self.depot = Depot(self.verlustTopf, self.taxRate)

    def test_purchase(self):
        self.depot.purchase(0, 100)
        self.depot.purchase(1, 200)
        self.depot.purchase(2, 300)

        positionsPurchaseValue_gt = np.array([100, 200, 300])
        np.testing.assert_array_equal(self.depot.positionsPurchaseValue, positionsPurchaseValue_gt)
        np.testing.assert_array_equal(self.depot.positionsCurrentValue, positionsPurchaseValue_gt)
        np.testing.assert_array_equal(self.depot.positionsPurchaseDate, [0,1,2])

    def test_taxes_scenario_1(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.9)
        self.depot.sell(90)
        self.assertEqual(self.depot.getCurrentTaxes(), 0)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(1.2)
        self.assertEqual(self.taxRate * 10, self.depot.getCurrentTaxes())
        cash = self.depot.sell(120)
        self.assertEqual(120 - 10 * self.taxRate, cash)


    def test_taxes_scenario_2(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.9)
        self.depot.sell(90)
        self.depot.purchase(0, 100)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(1.2)
        self.assertEqual(self.taxRate * 30, self.depot.getCurrentTaxes())
        self.assertEqual(200, self.depot.calculateSellAmount(193.93333333333334))
        cash = self.depot.sell(200)
        self.assertEqual(200 - 10 * self.taxRate - 80/120.0 * 20 * self.taxRate, cash)

    def test_taxes_scenario_3(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.assertEqual(150, self.verlustTopf.value)

        self.depot.purchase(0, 100)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(2.0)
        self.assertEqual(self.taxRate * 50, self.depot.getCurrentTaxes())
        self.assertEqual(400, self.depot.calculateSellAmount(387.0))
        cash = self.depot.sell(400)
        self.assertEqual(400 - 50 * self.taxRate, cash)

    def test_taxes_scenario_4(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.assertEqual(50, self.verlustTopf.value)

        self.depot.purchase(0, 100)
        self.depot.yieldInterest(2.0)
        self.assertEqual(self.taxRate * 50 , self.depot.getCurrentTaxes())
        self.assertEqual(50, self.depot.calculateSellAmount(50.0))
        cash = self.depot.sell(50)
        self.assertEqual(25, self.depot.verlustTopf.value)
        self.assertEqual(50 * self.taxRate, self.depot.getCurrentTaxes())
        self.assertEqual(50, cash)

    def test_taxes_scenario_5(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.depot.sell(50)
        self.assertEqual(150, self.verlustTopf.value)

        self.depot.purchase(0, 100)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(2.0)
        self.assertEqual(350, self.depot.calculateSellAmount(343.5))
        cash = self.depot.sell(350)
        self.assertEqual(0, self.depot.verlustTopf.value)
        self.assertEqual(25 * self.taxRate, self.depot.getCurrentTaxes())
        self.assertEqual(350 - 25 * self.taxRate, cash)

    def test_taxes_scenario_6(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(1.2)
        self.depot.sell(120)
        self.assertEqual(0, self.verlustTopf.value)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.9)
        cash = self.depot.sell(90)
        #within year negative verlustTopf not considered yet!
        self.assertEqual(10, self.verlustTopf.value)
        self.assertEqual(90, cash)

    def test_taxes_scenario_7(self):
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(4.0)
        self.depot.purchase(0, 100)
        self.depot.yieldInterest(0.5)
        self.assertEqual(50 * self.taxRate, self.depot.getCurrentTaxes())
        cash = self.depot.sell(250)
        #within transaction negative verlustTopf not considered yet!
        self.assertEqual(50, self.verlustTopf.value)
        self.assertEqual(250 - 100 * self.taxRate, cash)


if __name__ == '__main__':
    unittest.main()