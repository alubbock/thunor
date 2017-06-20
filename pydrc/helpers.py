import collections


def format_dose(num, sig_digits=12):
    """
    Formats a numeric dose like 1.2e-9 into 1.2 nM
    """
    if not isinstance(num, str) and isinstance(num, collections.Iterable):
        return [format_dose(each_num) for each_num in num]

    if num is None:
        return 'N/A'

    num = float(num)

    _prefix = {1e-12: 'p',
               1e-9: 'n',
               1e-6: 'μ',
               1e-3: 'm',
               1: ''}
    multiplier = 1
    for i in sorted(_prefix.keys()):
        if num >= i:
            multiplier = i
    return ('{0:.' + str(sig_digits) + 'g} {1}M').format(
        num/multiplier, _prefix[multiplier])
