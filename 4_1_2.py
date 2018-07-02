import tensorflow as tf
import csv

f = open('2015년.csv', 'r', encoding='euc-kr')
rdr = csv.reader(f)

y_data = []
x_data = []

i = -1;
for line in rdr:
    if i == -1:
        i = i + 1
    else:
        xx = []
        year = line[0][0:4]
        month = line[0][4:6]
        day = line[0][6:8]
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
        # xx.append(year)
        xx.append(month)
        xx.append(day)
        xx.append(dayOfWeekNum)

        x_data.append(xx)
        y_data.append(int(line[3]))
        i = i + 1

f.close()

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(x, w) + b;

cost = tf.reduce_mean(tf.square(hypothesis - y_data));

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session();

sess.run(tf.global_variables_initializer())
hhh = []
for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data})

    if step % 1000 == 0:
        # print(step, " Cost: ", cost_val, " prediction:", hy_val)
        hhh = hy_val

        i = 0
        costt = 0
        for yy in y_data:
            costt = costt + abs(yy - hhh[i])
            i = i + 1

        print("일당 오차율", costt / 365)
i = 0
cost = 0
for yy in y_data:
    cost = cost + abs(yy - hhh[i])
    i = i + 1

print("일당 오차율", cost/365)