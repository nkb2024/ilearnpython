from matplotlib import pyplot as plt
from collections import Counter
# # years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
# # gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# # # create a line chart, years on x-axis, gdp on y-axis
# # plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# # # add a title
# # plt.title("Nominal GDP")

# # # add a label to the y-axis
# # plt.ylabel("Billions of $")
# # plt.show()
# movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
# num_oscars = [5, 11, 3, 8, 10]

# # plot bars with left x-coordinates [0, 1, 2, 3, 4], heights [num_oscars]
# plt.bar(range(len(movies)), num_oscars)

# plt.title("My Favorite Movies")     # add a title
# plt.ylabel("# of Academy Awards")   # label the y-axis

# # label x-axis with movie names at bar centers
# plt.xticks(range(len(movies)), movies)

# plt.show()
# ---------------------------------------------

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# Bucket grades by decile, but put 100 in with the 90s
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # Shift bars right by 5
        histogram.values(),                 # Give each bar its correct height
        10,                                 # Give each bar a width of 10
        edgecolor=(0, 0, 0))                # Black edges for each bar

plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105,
                                           # y-axis from 0 to 5

plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()