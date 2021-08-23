import numpy as np

class VerlustTopf:
    def __init__(self):
        self.value = 0.0

class Depot:
    def __init__(self, verlustTopf, taxRate):
        self.positionsPurchaseDate = np.array([], dtype=np.float32)
        self.positionsPurchaseValue = np.array([], dtype=np.float32)
        self.positionsCurrentValue = np.array([], dtype=np.float32)

        self.taxRate = taxRate

        self.verlustTopf = verlustTopf

    def purchase(self, date, amount):
        self.positionsPurchaseValue = np.append(self.positionsPurchaseValue, amount)
        self.positionsCurrentValue = np.append(self.positionsCurrentValue, amount)
        self.positionsPurchaseDate = np.append(self.positionsPurchaseDate, date)

    def sell(self, amount):
        cash = 0
        stillToSell = amount
        i = 0
        for i in range(len(self.positionsPurchaseDate)):
            positionRemainder = self.positionsCurrentValue[i] - stillToSell
            positionGain = self.positionsCurrentValue[i] - self.positionsPurchaseValue[i]
            if positionRemainder <= 0:
                positionGainWithVerlusttopf = max(positionGain - self.verlustTopf.value, 0)
                stillToSell -= self.positionsCurrentValue[i]
                cash += self.positionsCurrentValue[i] - positionGainWithVerlusttopf * self.taxRate
                self.verlustTopf.value -= positionGain - positionGainWithVerlusttopf
                if stillToSell == 0:
                    i += 1
                    break
            else:
                stillToSellGain = positionGain * (stillToSell / self.positionsCurrentValue[i])
                stillToSellGainWithVerlusttopf = max((stillToSellGain - self.verlustTopf.value), 0)
                cash += stillToSell - self.taxRate * stillToSellGainWithVerlusttopf
                self.verlustTopf.value -= stillToSellGain - stillToSellGainWithVerlusttopf
                self.positionsPurchaseValue[i] *= positionRemainder / self.positionsCurrentValue[i]
                self.positionsCurrentValue[i] = positionRemainder
                break

        self.positionsCurrentValue = np.delete(self.positionsCurrentValue, list(range(i)))
        self.positionsPurchaseValue = np.delete(self.positionsPurchaseValue, list(range(i)))
        self.positionsPurchaseDate = np.delete(self.positionsPurchaseDate, list(range(i)))

        return cash

    def calculateSellAmount(self, cash):
        sellAmount = 0
        stillToCashIn = cash
        localVerlusttopf = self.verlustTopf.value
        for i in range(len(self.positionsPurchaseDate)):
            positionGain = self.positionsCurrentValue[i] - self.positionsPurchaseValue[i]
            positionGainWithVerlusttopf = max(positionGain - localVerlusttopf, 0)

            positionFullSellValue = self.positionsCurrentValue[i] - positionGainWithVerlusttopf * self.taxRate

            if positionFullSellValue <= stillToCashIn:
                stillToCashIn -= positionFullSellValue
                localVerlusttopf -= positionGain - positionGainWithVerlusttopf
                sellAmount +=  self.positionsCurrentValue[i]
            else:
                positionSellValue = (stillToCashIn - self.taxRate * localVerlusttopf) / (1 - self.taxRate * positionGain / self.positionsCurrentValue[i])
                positionPartGainWithVerlusttopf = (positionSellValue / self.positionsCurrentValue[i]) * positionGain - localVerlusttopf
                if positionPartGainWithVerlusttopf <= 0:
                    sellAmount += stillToCashIn
                else:
                    sellAmount += positionSellValue
                break

        return sellAmount

    def yieldInterest(self, interest):
        self.positionsCurrentValue *= interest

    def getCurrentValue(self):
        return np.sum(self.positionsCurrentValue)

    def getCurrentTaxes(self):
        return np.maximum(np.sum(self.positionsCurrentValue - self.positionsPurchaseValue) - self.verlustTopf.value, 0) * self.taxRate

    def getCurrentValueTaxed(self):
        return np.sum(self.positionsCurrentValue) - self.getCurrentTaxes()
