def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    # Write code here
    for i in range(order):
        new_series = []
        for l in range(len(series)+1):
            if l<len(series)-1:
                new_series.append(series[l+1] - series[l])
        series = new_series 

    return series
        