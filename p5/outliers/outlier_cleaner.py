#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = list()

    ### your code goes here

    # calculate errors in net_worths vs predictions
    errors = abs(net_worths - predictions)**2

    # combine input lists into single list
    single = zip(ages, net_worths, errors)

    # sort single list according to error from highest to lowest
    sorted_by_error = sorted(single, key=lambda x: x[2], reverse=True)

    # remove top 10 values
    cleaned_data = sorted_by_error[9:]

    return cleaned_data

