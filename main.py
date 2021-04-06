import math
import matplotlib.pyplot as plt


def fun(x):
    return math.cos(x)


def f_net(x, w, w0):
    net = w0
    for (k, j) in zip(x, w):
        net += k * j
    return net


class NeuralActivation:

    def __init__(self, window_size):
        self.a = 1
        self.b = 1.5
        self.c = 2 * self.b - self.a
        self.data = [self.a]
        self.window_size = window_size
        self.right_function_answer = [fun(self.a)]
        k = (self.b - self.a)/19
        a = self.a
        self.next_data = [self.a]
        for _ in range(0, 19):
            a += k
            self.right_function_answer.append(fun(a))
            self.data.append(a)
            self.next_data.append(a)

        while a <= self.c:
            a += k
            self.next_data.append(a)
            self.right_function_answer.append(fun(a))

    def go(self, max_epochs):
        answer_data = []
        w = [0]*(self.window_size + 1)
        epoch = 0
        nj = 0.8
        p = self.window_size
        sum_error = 1
        while epoch <= max_epochs and sum_error > 0.001:
            sum_error = 0
            y_net = []
            for i in range(p, len(self.data), 1):
                net = f_net(self.right_function_answer[i - p:i], w[1:], w[0])
                y_net.append(net)
                sigma = self.right_function_answer[i] - net
                sum_error += sigma*sigma

                for j in range(0, p):
                    w[j + 1] += sigma * nj * fun(self.right_function_answer[i - p + j])
            sum_error **= 0.5
            answer_data.append([w, sum_error, y_net])

            # print(epoch, sum_error, w)
            epoch += 1
        print('\n\n\nWINDOW SIZE = ', self.window_size, '\nEPOCHS = ', epoch - 1, '\n error = ', answer_data[-1][1], '\n'
              + ' weighs = ', answer_data[-1][0])
        self.print_answer(answer_data[-1], len(answer_data) - 1)

    def print_answer(self, answer_data, epochs):
        y = []
        y_w = self.right_function_answer[:self.window_size]

        for i in range(self.window_size, len(self.next_data)):
            y_w.append(f_net(self.right_function_answer[i - self.window_size:i], answer_data[0][1:], answer_data[0][0]))

        for i in self.next_data:
            y.append(fun(i))

        _, ax = plt.subplots()
        ax.plot(self.next_data, y, label='Реальная функция')
        ax.plot(self.next_data, y_w, label='Прогноз')
        ax.legend()
        plt.title(f'Окно размером: {self.window_size} | Количество эпох: {epochs}')  # заголовок
        plt.xlabel("x")  # ось абсцисс
        plt.ylabel("Реальная функция, Прогноз")  # ось ординат
        plt.grid()  # включение отображение сетки

        plt.show()


if __name__ == '__main__':
    for i in range(0, 3):
        n = NeuralActivation(6 + i)
        n.go(500)
        n.go(1000)

