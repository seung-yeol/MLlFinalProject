import tensorflow as tf
import matplotlib.pyplot as plt
import csv

y_data = []
x_data = []
days = []
def openCSV(fileName):
    global i
    f = open(fileName, 'r', encoding='euc-kr')
    rdr = csv.reader(f)
    i = -1
    for line in rdr:
        if i == -1:
            i = i + 1
        else:
            xx = []
            days.append(line[0])
            year = int(line[0][2:4])
            month = int(line[0][4:6])
            day = int(line[0][6:8])

            # xx.append(year)
            for i in range(12):
                if month == i + 1:
                    xx.append(1)
                else:
                    xx.append(0)

            for i in range(31):
                if day == i + 1:
                    xx.append(1)
                else:
                    xx.append(0)

            dayOfWeek = line[1]
            dayOfWeekNum = 0
            if dayOfWeek == "월":
                dayOfWeekNum = 1
            elif dayOfWeek == "화":
                dayOfWeekNum = 2
            elif dayOfWeek == "수":
                dayOfWeekNum = 3
            elif dayOfWeek == "목":
                dayOfWeekNum = 4
            elif dayOfWeek == "금":
                dayOfWeekNum = 5
            elif dayOfWeek == "토":
                dayOfWeekNum = 6
            elif dayOfWeek == "일":
                dayOfWeekNum = 7
            for j in range(7):
                if day == j + 1:
                    xx.append(1)
                else:
                    xx.append(0)

            season = 0
            if month == 12 or month == 1 or month == 2:
                season = 1
            elif month == 3 or month == 4 or month == 5:
                season = 2
            elif month == 6 or month == 7 or month == 8:
                season = 3
            elif month == 9 or month == 10 or month == 11:
                season = 4

            # xx.append(season)

            x_data.append(xx)
            y_data.append(int(line[2]))
            i = i + 1
    f.close()

openCSV("resource/2015년.csv")
openCSV("resource/2016년.csv")
openCSV("resource/2017년.csv")

x = tf.placeholder(tf.float32, shape=[None, 50])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([50, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

last = []
for step in range(4001):
    cost_val, hy_val, ww, _ = sess.run([cost, hypothesis, w, train], feed_dict={x: x_data})

    if step % 500 == 0:
        # print(step, " Cost: ", cost_val, " prediction:", hy_val)
        hhh = hy_val

        i = 0
        costt = 0
        for yy in y_data:
            costt = costt + abs(yy - hhh[i])
            i = i + 1

        print("일별 오차율", int(costt / (365*3)))
    if step == 4000:
        last = hy_val

x_data = []
y_data = []
days = []
openCSV("resource/2015년.csv")
openCSV("resource/2016년.csv")
openCSV("resource/2017년.csv")
openCSV("resource/2018년.csv")
hy_val = sess.run(hypothesis, feed_dict={x: x_data})
plt.plot(hy_val, 'r--', y_data, 'b--')
plt.show()

print("end")

