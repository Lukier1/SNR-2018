from bsif_filtering import procced_images
from learn import learnAndSaveModel
from result_stats import calcStatAndGenPlotForModel

print('Start descripting images')
# procced_images(3, 8)
# procced_images(7, 10)
# procced_images(11, 8)
# procced_images(13, 9)
# procced_images(15, 8)
print('... finish descripting images')

print('Start learning model and generating statistics ...')
learnAndSaveModel(3, 8, 5)
# learnAndSaveModel(3, 8, 4)
# learnAndSaveModel(3, 8, 3)
# learnAndSaveModel(3, 8, 2)
# learnAndSaveModel(3, 8, 1)

calcStatAndGenPlotForModel(3, 8, 5)
# calcStatAndGenPlotForModel(3, 8, 4)
# calcStatAndGenPlotForModel(3, 8, 3)
# calcStatAndGenPlotForModel(3, 8, 2)
# calcStatAndGenPlotForModel(3, 8, 1)

####################################################

# learnAndSaveModel(7, 10, 5)
# learnAndSaveModel(7, 10, 4)
# learnAndSaveModel(7, 10, 3)
# learnAndSaveModel(7, 10, 2)
# learnAndSaveModel(7, 10, 1)

# calcStatAndGenPlotForModel(7, 10, 5)
# calcStatAndGenPlotForModel(7, 10, 4)
# calcStatAndGenPlotForModel(7, 10, 3)
# calcStatAndGenPlotForModel(7, 10, 2)
# calcStatAndGenPlotForModel(7, 10, 1)


# learnAndSaveModel(11, 8, 5)
# learnAndSaveModel(11, 8, 4)
# learnAndSaveModel(11, 8, 3)
# learnAndSaveModel(11, 8, 2)
# learnAndSaveModel(11, 8, 1)

# calcStatAndGenPlotForModel(11, 8, 5)
# calcStatAndGenPlotForModel(11, 8, 4)
# calcStatAndGenPlotForModel(11, 8, 3)
# calcStatAndGenPlotForModel(11, 8, 2)
# calcStatAndGenPlotForModel(11, 8, 1)


# learnAndSaveModel(13, 9, 5)
# learnAndSaveModel(13, 9, 4)
# learnAndSaveModel(13, 9, 3)
# learnAndSaveModel(13, 9, 2)
# learnAndSaveModel(13, 9, 1)

# calcStatAndGenPlotForModel(13, 9, 5)
# calcStatAndGenPlotForModel(13, 9, 4)
# calcStatAndGenPlotForModel(13, 9, 3)
# calcStatAndGenPlotForModel(13, 9, 2)
# calcStatAndGenPlotForModel(13, 9, 1)

# learnAndSaveModel(15, 8, 5)
# learnAndSaveModel(15, 8, 4)
# learnAndSaveModel(15, 8, 3)
# learnAndSaveModel(15, 8, 2)
# learnAndSaveModel(15, 8, 1)

# calcStatAndGenPlotForModel(15, 8, 5)
# calcStatAndGenPlotForModel(15, 8, 4)
# calcStatAndGenPlotForModel(15, 8, 3)
# calcStatAndGenPlotForModel(15, 8, 2)
# calcStatAndGenPlotForModel(15, 8, 1)

print('.. end learning model and generating statistics')