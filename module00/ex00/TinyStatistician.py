import unittest
from typing import List, Union, Optional


class TinyStatistician:

    @staticmethod
    def __is_invalid_input(x: List[Union[int, float]]) -> bool:
        if not isinstance(x, list):
            return True
        elif not all(isinstance(i, (int, float)) for i in x):
            return True
        elif len(x) == 0:
            return True
        return False

    @staticmethod
    def mean(x: List[Union[int, float]]) -> Optional[float]:
        """
        Computes the mean of a given non-empty list of int or float.

        Args:
            x: A list of integers or floats.

        Returns:
            The mean as a float, or None if input is invalid.
        """
        if TinyStatistician.__is_invalid_input(x):
            return None
        total = 0.0
        for i in x:
            total += i
        return total / len(x)

    @staticmethod
    def median(x: List[Union[int, float]]) -> Optional[float]:
        """
        Computes the median of a given non-empty list of int or float.

        Args:
            x: A list of integers or floats.

        Returns:
            The median as a float, or None if input is invalid.
        """
        if TinyStatistician.__is_invalid_input(x):
            return None
        sorted_x = sorted(x)
        n = len(sorted_x)
        middle = n // 2
        if n % 2 == 0:
            return (sorted_x[middle - 1] + sorted_x[middle]) / 2
        else:
            return float(sorted_x[middle])

    @staticmethod
    def quartile(x: List[Union[int, float]]) -> Optional[List[float]]:
        """
        Computes the quartiles Q1 and Q3 of a given non-empty list of
        int or float.

        Args:
            x: A list of integers or floats.

        Returns:
            A list containing Q1 and Q3 as floats, or None if input is invalid.
        """
        if TinyStatistician.__is_invalid_input(x):
            return None
        sorted_x = sorted(x)
        n = len(sorted_x)
        middle = n // 2
        q1_index = middle // 2
        q3_index = middle + q1_index
        if n % 2 == 0:
            q1 = (sorted_x[q1_index - 1] + sorted_x[q1_index]) / 2
            q3 = (sorted_x[q3_index - 1] + sorted_x[q3_index]) / 2
        else:
            q1 = sorted_x[q1_index]
            q3 = sorted_x[q3_index]
        return [float(q1), float(q3)]

    @staticmethod
    def percentile(x: List[Union[int, float]], percentile: int) -> Optional[float]:
        """
        Computes the percentile p of a given non-empty list of int or float.
        Note: Uses linear interpolation between the two closest list elements.

        Args:
            x: A list of integers or floats.
            percentile: An integer between 0 and 100.

        Returns:
            The percentile value as a float, or None if input is invalid.
        """
        if TinyStatistician.__is_invalid_input(x):
            return None
        elif not isinstance(percentile, int):
            return None
        elif percentile < 0 or percentile > 100:
            return None
        sorted_x = sorted(x)
        n = len(sorted_x)
        if percentile == 0:
            return float(sorted_x[0])
        if percentile == 100:
            return float(sorted_x[-1])
        index = (percentile / 100) * (n - 1)
        floor_index = int(index)
        frac = index - floor_index
        return sorted_x[floor_index] + frac * (sorted_x[floor_index + 1] - sorted_x[floor_index])

    @staticmethod
    def var(x: List[Union[int, float]]) -> Optional[float]:
        """
        Computes the variance of a given non-empty list of int or float.
        Note: Uses the unbiased estimator (divides by n - 1).

        Args:
            x: A list of integers or floats.

        Returns:
            The variance as a float, or None if input is invalid.
        """
        if TinyStatistician.__is_invalid_input(x):
            return None
        mean_val = TinyStatistician.mean(x)
        if mean_val is None:
            return None
        diff = sum((i - mean_val) ** 2 for i in x)
        return diff / (len(x) - 1)

    @staticmethod
    def std(x: List[Union[int, float]]) -> Optional[float]:
        """
        Computes the standard deviation of a given non-empty list of
        int or float.

        Args:
            x: A list of integers or floats.

        Returns:
            The standard deviation as a float, or None if input is invalid.
        """
        var_val = TinyStatistician.var(x)
        if var_val is None:
            return None
        return var_val ** 0.5


class TestTinyStatistician(unittest.TestCase):

    def test_mean(self):
        self.assertAlmostEqual(
            TinyStatistician.mean([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(
            TinyStatistician.mean([1.5, 2.5, 3.5, 4.5, 5.5]), 3.5)
        self.assertAlmostEqual(
            TinyStatistician.mean([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            TinyStatistician.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 4.0)
        self.assertAlmostEqual(
            TinyStatistician.mean([1, 2, 3, 4, 5, 6, 7]), 4.0)
        self.assertAlmostEqual(
            TinyStatistician.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 4.5)
        self.assertAlmostEqual(
            TinyStatistician.mean([1, 2, 3, 4, 5, 6, 7, 8]), 4.5)
        self.assertAlmostEqual(
            TinyStatistician.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), 5.0)
        self.assertAlmostEqual(
            TinyStatistician.mean([1, 2, 3, 4, 5, 6, 7, 8, 9]), 5.0)
        self.assertAlmostEqual(
            TinyStatistician.mean([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]), 5.5)

    def test_mean_with_invalid_input(self):
        self.assertIsNone(TinyStatistician.mean([]))
        self.assertIsNone(TinyStatistician.mean([1, 2, 'a', 4, 5]))
        self.assertIsNone(TinyStatistician.mean('1, 2, 3, 4, 5'))

    def test_median(self):
        self.assertAlmostEqual(
            TinyStatistician.median([1, 2, 3, 4, 5]), 3.0)
        self.assertAlmostEqual(
            TinyStatistician.median([1.5, 2.5, 3.5, 4.5, 5.5]), 3.5)
        self.assertAlmostEqual(
            TinyStatistician.median([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            TinyStatistician.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 4.0)
        self.assertAlmostEqual(
            TinyStatistician.median([1, 2, 3, 4, 5, 6, 7]), 4.0)
        self.assertAlmostEqual(
            TinyStatistician.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 4.5)
        self.assertAlmostEqual(
            TinyStatistician.median([1, 2, 3, 4, 5, 6, 7, 8]), 4.5)
        self.assertAlmostEqual(
            TinyStatistician.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]), 5.0)
        self.assertAlmostEqual(
            TinyStatistician.median([1, 2, 3, 4, 5, 6, 7, 8, 9]), 5.0)
        self.assertAlmostEqual(
            TinyStatistician.median([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]), 5.5)

    def test_median_with_invalid_input(self):
        self.assertIsNone(TinyStatistician.median([]))
        self.assertIsNone(TinyStatistician.median([1, 2, 'a', 4, 5]))
        self.assertIsNone(TinyStatistician.median('1, 2, 3, 4, 5'))

    def test_quartile(self):
        self.assertEqual(
            TinyStatistician.quartile([1, 42, 300, 10, 59]), [10.0, 59.0])
        self.assertEqual(
            TinyStatistician.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9]), [3.0, 7.0])
        self.assertEqual(
            TinyStatistician.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [2.5, 7.5])
        self.assertEqual(
            TinyStatistician.quartile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), [3.0, 8.0])

    def test_quartile_with_invalid_input(self):
        self.assertIsNone(TinyStatistician.quartile([]))
        self.assertIsNone(TinyStatistician.quartile([1, 2, 'a', 4, 5]))
        self.assertIsNone(TinyStatistician.quartile('1, 2, 3, 4, 5'))

    def test_percentile(self):
        self.assertAlmostEqual(
            TinyStatistician.percentile([1, 2, 3, 4, 5], 10), 1.4)
        self.assertAlmostEqual(
            TinyStatistician.percentile([1, 2, 3, 4, 5, 6], 10), 1.5)
        self.assertAlmostEqual(
            TinyStatistician.percentile([1, 2, 3, 4, 5, 6, 7], 10), 1.6)
        self.assertAlmostEqual(
            TinyStatistician.percentile([1, 2, 3, 4, 5, 6, 7, 8], 10), 1.7)
        self.assertAlmostEqual(
            TinyStatistician.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9], 10), 1.8)

    def test_percentile_with_invalid_input(self):
        self.assertIsNone(
            TinyStatistician.percentile([], 10))
        self.assertIsNone(
            TinyStatistician.percentile([1, 2, 'a', 4, 5], 10))
        self.assertIsNone(
            TinyStatistician.percentile('1, 2, 3, 4, 5', 10))
        self.assertIsNone(
            TinyStatistician.percentile([1, 2, 3, 4, 5], 'a'))
        self.assertIsNone(
            TinyStatistician.percentile([1, 2, 3, 4, 5], -10))
        self.assertIsNone(
            TinyStatistician.percentile([1, 2, 3, 4, 5], 110))

    def test_var(self):
        self.assertAlmostEqual(
            TinyStatistician.var([1, 2, 3, 4, 5]), 2.5)
        self.assertAlmostEqual(
            TinyStatistician.var([1.5, 2.5, 3.5, 4.5, 5.5]), 2.5)
        self.assertAlmostEqual(
            TinyStatistician.var([1, 2, 3, 4, 5, 6]), 3.5)
        self.assertAlmostEqual(
            TinyStatistician.var([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 3.5)

    def test_var_with_invalid_input(self):
        self.assertIsNone(TinyStatistician.var([]))
        self.assertIsNone(TinyStatistician.var([1, 2, 'a', 4, 5]))
        self.assertIsNone(TinyStatistician.var('1, 2, 3, 4, 5'))

    def test_std(self):
        self.assertAlmostEqual(
            TinyStatistician.std([1, 2, 3, 4, 5]), 1.5811, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1.5, 2.5, 3.5, 4.5, 5.5]), 1.5811, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1, 2, 3, 4, 5, 6]), 1.8708, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 1.8708, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1, 2, 3, 4, 5, 6, 7]), 2.1602, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]), 2.1602, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1, 2, 3, 4, 5, 6, 7, 8]), 2.4495, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
            2.4495, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1, 2, 3, 4, 5, 6, 7, 8, 9]), 2.7386, places=4)
        self.assertAlmostEqual(
            TinyStatistician.std([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
            2.7386, places=4)

    def test_std_with_invalid_input(self):
        self.assertIsNone(TinyStatistician.std([]))
        self.assertIsNone(TinyStatistician.std([1, 2, 'a', 4, 5]))
        self.assertIsNone(TinyStatistician.std('1, 2, 3, 4, 5'))


if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    print(TinyStatistician.mean(a))
    # Output:
    # 82.4
    print(TinyStatistician.median(a))
    # Output:
    # 42.0
    print(TinyStatistician().quartile(a))
    # Output:
    # [10.0, 59.0]
    print(TinyStatistician().percentile(a, 10))
    # Output:
    # 4.6
    print(TinyStatistician().percentile(a, 15))
    # Output:
    # 6.4
    print(TinyStatistician().percentile(a, 20))
    # Output:
    # 8.2
    print(TinyStatistician().var(a))
    # Output:
    # 15349.3
    print(TinyStatistician().std(a))
    # Output:
    # 123.89229193133849

    unittest.main()
