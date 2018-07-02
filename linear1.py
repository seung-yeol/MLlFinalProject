import tensorflow as tf
import csv

y_data = []
x_data = []

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
            year = line[0][2:4]
            month = line[0][4:6]
            day = line[0][6:8]
            xx.append(year)
            xx.append(month)
            xx.append(day)

            x_data.append(xx)
            y_data.append(int(line[2]))
            i = i + 1
    f.close()

openCSV("resource/2015년.csv")
openCSV("resource/2016년.csv")
openCSV("resource/2017년.csv")

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
hhh = []
for step in range(4001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x: x_data})

    if step % 500 == 0:
        # print(step, " Cost: ", cost_val, " prediction:", hy_val)
        hhh = hy_val

        i = 0
        costt = 0
        for yy in y_data:
            costt = costt + abs(yy - hhh[i])
            i = i + 1

        print("일별 오차수", int(costt / (365*3)))
