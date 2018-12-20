from result_stats import calc_stat_and_gen_plot_for_model

if __name__ == '__main__':
    print('Start descripting images')
    # procced_images(3, 8)
    # procced_images(7, 10)
    # procced_images(11, 8)
    # procced_images(13, 9)
    # procced_images(15, 8)
    print('... finish descripting images')

    print('Start learning model and generating statistics ...')
    # learnAndSaveModel(3, 8, 5)
    # learnAndSaveModel(3, 8, 4)
    # learnAndSaveModel(3, 8, 3)
    # learnAndSaveModel(3, 8, 2)
    # learnAndSaveModel(3, 8, 1)

    calc_stat_and_gen_plot_for_model(3, 8, 5)
    calc_stat_and_gen_plot_for_model(3, 8, 4)
    calc_stat_and_gen_plot_for_model(3, 8, 3)
    calc_stat_and_gen_plot_for_model(3, 8, 2)
    calc_stat_and_gen_plot_for_model(3, 8, 1)

    ####################################################

    # learnAndSaveModel(7, 10, 5)
    # learnAndSaveModel(7, 10, 4)
    # learnAndSaveModel(7, 10, 3)
    # learnAndSaveModel(7, 10, 2)
    # learnAndSaveModel(7, 10, 1)

    calc_stat_and_gen_plot_for_model(7, 10, 5)
    calc_stat_and_gen_plot_for_model(7, 10, 4)
    calc_stat_and_gen_plot_for_model(7, 10, 3)
    calc_stat_and_gen_plot_for_model(7, 10, 2)
    calc_stat_and_gen_plot_for_model(7, 10, 1)

    # learnAndSaveModel(11, 8, 5)
    # learnAndSaveModel(11, 8, 4)
    # learnAndSaveModel(11, 8, 3)
    # learnAndSaveModel(11, 8, 2)
    # learnAndSaveModel(11, 8, 1)

    calc_stat_and_gen_plot_for_model(11, 8, 5)
    calc_stat_and_gen_plot_for_model(11, 8, 4)
    calc_stat_and_gen_plot_for_model(11, 8, 3)
    calc_stat_and_gen_plot_for_model(11, 8, 2)
    calc_stat_and_gen_plot_for_model(11, 8, 1)

    # learnAndSaveModel(13, 9, 5)
    # learnAndSaveModel(13, 9, 4)
    # learnAndSaveModel(13, 9, 3)
    # learnAndSaveModel(13, 9, 2)
    # learnAndSaveModel(13, 9, 1)

    calc_stat_and_gen_plot_for_model(13, 9, 5)
    calc_stat_and_gen_plot_for_model(13, 9, 4)
    calc_stat_and_gen_plot_for_model(13, 9, 3)
    calc_stat_and_gen_plot_for_model(13, 9, 2)
    calc_stat_and_gen_plot_for_model(13, 9, 1)

    # learnAndSaveModel(15, 8, 5)
    # learnAndSaveModel(15, 8, 4)
    # learnAndSaveModel(15, 8, 3)
    # learnAndSaveModel(15, 8, 2)
    # learnAndSaveModel(15, 8, 1)

    calc_stat_and_gen_plot_for_model(15, 8, 5)
    calc_stat_and_gen_plot_for_model(15, 8, 4)
    calc_stat_and_gen_plot_for_model(15, 8, 3)
    calc_stat_and_gen_plot_for_model(15, 8, 2)
    calc_stat_and_gen_plot_for_model(15, 8, 1)

    print('.. end learning model and generating statistics')
