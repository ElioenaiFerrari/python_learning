from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class MachineModel:
    _model = LinearSVC()

    def trainer(self, trainer_x, trainer_y):
        self._model.fit(trainer_x, trainer_y)

    def predict(self, items):
        return self._model.predict([items])

    def accuracy(self, test_x, test_y):
        predicts = self._model.predict(test_x)

        return accuracy_score(test_y, predicts)
