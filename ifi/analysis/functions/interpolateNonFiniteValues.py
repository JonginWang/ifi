#!/usr/bin/env python3
"""
signalProcessing
Copyright (C) 2024  Dominik Rueß

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------

interpolation method for signals, both scattered and uniformly spaced, with non-finite values
such as nan of inf

also includes unit tests for some boundary and general use case scenarios, run as:
    python3 interpolateNonFiniteValues.py
    
requires third party modlues numpy and scipy

features:
- allow / disallow extrapolation 
- limit the number of non-finite values next to each vlaues (otherwise leave as nan/inf)
    (this somehow implicitly expects ordered X-Values, if given)
- different spline degree
- allow scattered data position (x coords)

@date: 2024-02-09
@author: Dominik Rueß
@copyright: © 2024, Dominik Rueß
@licence: GPL v3

Revised by: Jongin Wang
@date: 2025-12-03

Modifications:
- Fixed interpolation method selection logic based on smoothing parameter (s):
  * If s == 0: Uses scipy.interpolate.interp1d with spline interpolation (no smoothing)
    - Supports all degrees (0: 'zero', 1: 'slinear', 2: 'quadratic', 3: 'cubic')
    - Cannot use splrep with s = 0, so interp1d wrapper is used
  * If s > 0: Uses scipy.interpolate.splrep (spline representation with smoothing)
    - Supports degrees 1, 2, 3 (smoothing splines)
    - The smoothing parameter s controls the trade-off between smoothness and fit quality

"""

# sys imports
import unittest
import warnings
import math

# third party imports
import scipy.interpolate
import numpy as np

def find_nearest(array,value):
    """
    https://stackoverflow.com/a/26026189/1150303
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def interpolateNonFinite(   signal : np.array, 
                            xCoords = None, 
                            maxNonFiniteNeighbors = -1, 
                            degree = 3,
                            allowExtrapolation = False,
                            smooth = 0) -> np.array:
    """
    assume uniform spacing if xCoords is None
    
    interpolate nan values in a signal, if enough data is present
    
    @param maxNonFiniteNeighbors: the maximum number of non-finite neighbors for the current value to
             be interpolated (otherwise it stays non-finite). Not used for <= 0
    @param degree: interpolation spline degree, choice of 0, 1, 2 or 3
    @param xCoords: if the signal values are non-uniformly spaced   
    @param smooth: either "auto", 0 or any positive value.
             Determines interpolation method:
             - If s == 0: Uses scipy.interpolate.interp1d (no smoothing, exact fit)
             - If s > 0: Uses scipy.interpolate.splrep (smoothing spline, s controls smoothness)
             - "auto": s = n - sqrt(2*n) where n is number of finite points
    @returns: numpy array of float type. Note not all non-finite values may be interpolated, mainly
              depending on the maxNonFiniteNeighbors parameter
    
    """
    if degree < 0 or degree > 3 or type(degree) is not int:
        raise AssertionError("spline degree needs to be in [0, 1, 2, 3]")
    
    lSignal = len(signal)
    if lSignal == 0:
        return np.array(signal).astype(float)
        
    data = np.array(signal).astype(float).flatten()
        
    if xCoords is not None:
        if len(signal) != len(xCoords):
            raise AssertionError("signal and x coordinates need to have same size")
            
        xCoords = np.array(xCoords).flatten()
        
        # make sure xCoords are unique, otherwise scipy interpolation behaviour is not defined
        if len(np.unique(xCoords)) != len(xCoords):
            raise AssertionError("there must no be identical values within the x coords:"
                                 " the provided (scattered) data positions must be unique.")
                                 
        if maxNonFiniteNeighbors > 0:
            sortedXValues = sorted(xCoords)
            if (sortedXValues != xCoords).any():
                raise AssertionError("for the feature of 'maxNonFiniteNeighbors', the "
                                     "xCoords array needs to be sorted")
            
    else:
        # assume uniform spacing in [0, 1)
        xCoords = np.array(range(lSignal)) / lSignal
                        
    
    # find non-nan and non-inf values
    finiteIndexes = np.where(np.isfinite(data))[0]
    lFinite = len(finiteIndexes)
    
    if lFinite == lSignal:
        # no need to interpolate
        return np.array(signal).astype(float)
        
    elif lFinite == 0:
        raise AssertionError("no finite data found (all values are inf or nan)")        
    #elif lFinite == 1 and lSignal > 1 and degree == 0: 
    #    warnings.warn("Only one finite data point found, all interpolations will be constant", RuntimeWarning)
    #    return np.ones( lSignal ) * data[finiteIndexes[0]]
        
    elif lFinite <= degree:
        raise AssertionError(f"for degree {degree}, at least {degree+1} finite points are required")
    
    # generate the interpolation class 
    
    if smooth == "auto":                                               
        s = lFinite - np.sqrt(2 * lFinite)
    elif smooth is None:
        s = 0
    else:
        s = np.abs(float(smooth))
        
    if degree == 0:
        kind = 'zero'
    elif degree == 1:
        kind = 'slinear'
    elif degree == 2:   
        kind = 'quadratic'
    elif degree == 3: 
        kind = 'cubic'

    if s == 0:
        if degree > 0:
            warnings.warn("degree 0 (constant) interpolation cannot be used with smoothing", RuntimeWarning)
            
        # For any degree, if s == 0, use interp1d for being unable to use splrep with s = 0
        interpolation = scipy.interpolate.interp1d(
            xCoords[finiteIndexes],
            data[finiteIndexes],
            kind=kind,
            bounds_error=not allowExtrapolation,
            fill_value="extrapolate" if allowExtrapolation else 0.0
        )
        useWrapper = True
    else:
        # For any degree, if s > 0, use splrep (spline representation)
        useWrapper = False
        interpolation = scipy.interpolate.splrep(
            xCoords[finiteIndexes],
            data[finiteIndexes],
            k=degree,
            s=s
        )

    # interpolate only at the values of the original input without finite value 
    # i.e. leave original finite signal values as given
    interpolateAtIndexes = np.setxor1d(range(lSignal), finiteIndexes)

    # do not extrapolate values which extend the first and last finite value
    if not allowExtrapolation:
        first = finiteIndexes[0]
        last = finiteIndexes[-1]
        interpolateAtIndexes = [x for x in interpolateAtIndexes if x > first and x < last]
        
    if maxNonFiniteNeighbors > 0:
        interpolateAtIndexes = [x for x in interpolateAtIndexes 
                                if abs(find_nearest(finiteIndexes, x) - x) <= maxNonFiniteNeighbors]
                                
    if len(interpolateAtIndexes) == 0:
        return data
        
    if useWrapper:
        data[interpolateAtIndexes] = interpolation(xCoords[interpolateAtIndexes])
    else:
        data[interpolateAtIndexes] = scipy.interpolate.splev(xCoords[interpolateAtIndexes], interpolation)
        
    return data

class TestinterpolateNonFinite(unittest.TestCase):

    def maxDifference(firstArray, secondArray):
        return np.max(np.abs(firstArray - secondArray))

    def test_length(self):
        """
        test the comparison of the input length, for given xCoords
        """
        
        # different lengths
        with self.assertRaises(AssertionError):
            interpolateNonFinite([1,0], xCoords = [0])
            interpolateNonFinite([], xCoords = [0])
            interpolateNonFinite([1,0], xCoords = [])
            
        # same lengths
        try:
            interpolateNonFinite([1,0], xCoords = [1, 0])
            interpolateNonFinite([], xCoords = [])
        except AssertionError:
            self.fail("interpolateNonFinite() raised AssertionError unexpectedly!")
            
            
        # xcoords need to be unique
        with self.assertRaises(AssertionError): 
            interpolateNonFinite([1,0], xCoords = [0, 0])
            interpolateNonFinite([1,0, 1], xCoords = [0., 0, 1])
            

    def test_extremes(self):
        """
        test some extreme inputs, as only one finite value or only non-finite values
        """
        
       
        with self.assertRaises(AssertionError):
            interpolateNonFinite([np.nan])
            interpolateNonFinite([np.inf])
            interpolateNonFinite([np.nan, np.inf])    
              
    def test_degreeWrong(self):
        """
        test some wronge degree inputs
        """
        with self.assertRaises(AssertionError):
            interpolateNonFinite([], degree = -1)
            interpolateNonFinite([], degree = 4)
            interpolateNonFinite([], degree = 1.)
            interpolateNonFinite([], degree = 1.3)
            interpolateNonFinite([], degree = "test")
         
    def test_smallSizes(self):
        """
        test some small input sizes
        """
    
        # no interpolation required test:
        self.assertEqual(interpolateNonFinite([1]).tolist(),  [1.])
        self.assertEqual(interpolateNonFinite([1.]).tolist(), [1.])
        self.assertEqual(interpolateNonFinite([1,  0]).tolist(),  [1., 0.])
        self.assertEqual(interpolateNonFinite([1., 0.]).tolist(), [1., 0.])
        
    def test_linear(self):
        """
        test the filling of some linear graphs
        """
        
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(interpolateNonFinite([0, np.nan, 2], degree = 1), 
                                                  [0., 1., 2.]), 
                               0)
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(interpolateNonFinite([0, np.nan, np.inf, 3], degree = 1), 
                                                  [0., 1., 2., 3.]), 
                               0)
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(interpolateNonFinite([2, np.inf, 0], degree = 1), 
                                                  [2., 1., 0.]), 
                               0)
                               
    def test_extrapolation(self):
        """
        test extrapolation and non-extrapolation as given by the respective function parameter
        """
        
        # disallow extrapolation by default:
        r = interpolateNonFinite([np.nan, 0, np.nan, 2, np.nan], degree = 1)
        self.assertTrue(np.isnan(r[0]))
        self.assertTrue(np.isnan(r[-1]))
        
        # the interpolated values still need to be correct:
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(r[1:-1], [0., 1., 2.]), 0)
        
        # allow extrapolation now
        r = interpolateNonFinite([np.nan, 0, np.nan, 2, np.nan], degree = 1, allowExtrapolation = True)
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(r, [-1., 0., 1., 2., 3.]), 0)
        
    def test_scatteredData(self):
        """
        test on non-uniform x values
        """

        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(
                                    interpolateNonFinite([0, np.nan, 20], xCoords = [0., 1., 20.], degree = 1), 
                                                  [0., 1., 20.]), 
                               0)
                               
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(
                                    interpolateNonFinite([0, np.nan, 20], xCoords = [0., 5., 20.], degree = 1), 
                                                  [0., 5., 20.]), 
                               0)
                               
                               
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(
                                    interpolateNonFinite([np.nan, 0, np.nan, 20], xCoords = [-5., 0., -1., 20.],
                                                         degree = 1, allowExtrapolation = True), 
                                                  [-5., 0., -1., 20.]), 
                               0)
                               
    def test_sinus(self):
        """
        test the filling of non-linear sinus graph
        """
        x = np.linspace(0, np.pi, 1000)
        yOrig = np.sin(x)
        y = yOrig.copy()
        y[::10] = np.nan
        
        yFilled = interpolateNonFinite(y, xCoords = x, allowExtrapolation = True)
        
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(yFilled, yOrig), 0)
        
    def test_minLength(self):
        """
        test the minimum number of points
        """
        
        # test not enough points
        with self.assertRaises(AssertionError):
            interpolateNonFinite([np.nan], degree = 0)
            interpolateNonFinite([1, np.nan], degree = 1)
            interpolateNonFinite([1, np.nan], degree = 2)
            interpolateNonFinite([1, 2., np.nan], degree = 2)
            interpolateNonFinite([1, np.nan], degree = 3)
            interpolateNonFinite([1, 2., np.nan], degree = 3)
            interpolateNonFinite([1, 2., 3., np.nan], degree = 3)
    
        # test enough points
        try:
            interpolateNonFinite([1, np.nan], degree = 0)
            interpolateNonFinite([1, 2., np.nan], degree = 1)
            interpolateNonFinite([1, 2., 3., np.nan], degree = 2)
            interpolateNonFinite([1, 2., 3., 4, np.nan], degree = 3)
        except AssertionError:
            self.fail("interpolateNonFinite() raised AssertionError unexpectedly!")
              
        
    def test_maxNonFiniteNeighbors(self):
        """
        test the maximum number of non-finite neighbours being filled
        """
        y = np.array([0., np.nan, np.nan, np.nan, np.nan, np.nan, 6.])
        
        # all values are interpolated
        yFilled = interpolateNonFinite(y, maxNonFiniteNeighbors = 0, degree =1)
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(yFilled, range(0, 7)), 0)
        
        # test different numNeighbors
        yFilled = interpolateNonFinite(y, maxNonFiniteNeighbors = 1, degree =1)
        for ind in [2, 3, 4]:
            self.assertTrue(np.isnan(yFilled[ind]))
        yFilled = interpolateNonFinite(y, maxNonFiniteNeighbors = 2, degree =1)
        for ind in [3]:
            self.assertTrue(np.isnan(yFilled[ind]))
        yFilled = interpolateNonFinite(y, maxNonFiniteNeighbors = 3, degree =1)
        self.assertAlmostEqual(TestinterpolateNonFinite.maxDifference(yFilled, range(0, 7)), 0)
        
        # test not sorted / sorted
        with self.assertRaises(AssertionError):
            yFilled = interpolateNonFinite(y, xCoords = np.array(range(len(y)))[::-1],
                                           maxNonFiniteNeighbors = 3, degree =1)
        try:
            yFilled = interpolateNonFinite(y, xCoords = np.array(range(len(y))),
                                           maxNonFiniteNeighbors = 3, degree =1)
        except AssertionError:
            self.fail("interpolateNonFinite() raised AssertionError unexpectedly!")
        

if __name__ == '__main__':
    unittest.main()