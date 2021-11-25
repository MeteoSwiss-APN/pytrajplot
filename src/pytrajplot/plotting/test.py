"""TEST FUNCTION."""


def is_visible(lat, lon, domain_boundaries, cross_dateline) -> bool:
    """Summary - First line should end with a period.

    Args:
        lat
                            type       description
        lon
                            type       description
        domain_boundaries
                            type       description
        cross_dateline
                            type       description

    Returns:
        output_variable     type       description

    """
    if cross_dateline:
        if lon < 0:
            lon = 360 - abs(lon)

    in_domain = (
        domain_boundaries[0] <= float(lon) <= domain_boundaries[1]
        and domain_boundaries[2] <= float(lat) <= domain_boundaries[3]
    )

    if in_domain:
        return True, False
    else:
        return False, False
