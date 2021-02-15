function [Trend, Cyclical] = hpfilter(Series, Smoothing)
%HPFILTER Hodrick-Prescott filter
% Separate one or more time series into trend and cyclical components with the
% Hodrick-Prescott filter. If no output arguments specified, display a plot of
% the series and trend (with cycles removed). The plot option can be used to
% help select a Smoothing parameter.
%
%	hpfilter(Series);
%	hpfilter(Series, Smoothing);
%	Trend = hpfilter(Series, Smoothing);
%	[Trend, Cyclical] = hpfilter(Series, Smoothing);
%
% Inputs:
%	Series - NUMSAMPLES x NUMSERIES matrix of NUMSERIES time series with
%		NUMSAMPLES samples.
%
% Optional Inputs:
%	Smoothing - Either a scalar single value to be applied to all
%		series or a NUMSERIES-vector with values to be applied to each series.
%		The default value is 1600, which is suggested in the reference as a
%		value for quarterly data. If the smoothing parameter is 0, no smoothing
%		is done. As the smoothing parameter increases in value, the smoothed
%		series becomes a straight line. If the smoothing parameter is Inf
%		(which is permitted), the series is detrended.
%
% Outputs:
%	Trend - Trend component of Series.
%	Cyclical - Cyclical component of Series.
%
% Notes:
%	The Hodrick-Prescott filter separates a time series into a "trend"
%	component and a "cyclical" component such that
%		Series = Trend + Cyclical
%	and is equivalent to a cubic spline smoother, where the smoothed portion is
%	in the Trend.
%
%	The reference suggests values for the smoothing parameter that depend upon
%	the periodicity of the data with:
%		Periodicity		Smoothing
%		Yearly			100
%		Quarterly		1600
%		Monthly			14400
%
% WARNING:
%	The Hodrick-Prescott filter can produce anomalous endpoint effects in very
%	high-frequency data and should never be used for extrapolation.
%
% References:
%	[1] Robert J. Hodrick and Edward C. Prescott, "Postwar U.S. Business Cycles:
%		An Empircal Investigation," Journal of Money, Credit, and Banking,
%		Vol. 29, No. 1, February 1997, pp. 1-16.

%	Copyright 1999-2006 The MathWorks, Inc.
%	$Revision: 1.1.8.1 $   $Date: 2008/04/18 21:15:48 $

% Step 1 - check arguments

if nargin < 1 || isempty(Series)
	error('econ:hpfilter:MissingInputData', ...
		'Required times series data Series missing or empty.');
end

if ~isscalar(Series) && isvector(Series) && isa(Series,'double')
	Series = Series(:);
	[NumSamples, NumSeries] = size(Series);
elseif ndims(Series) == 2 && min(size(Series)) > 1 && isa(Series,'double')
	[NumSamples, NumSeries] = size(Series);
else
	error('econ:hpfilter:InvalidInputArg', ...
 		'Invalid format for time series data Series. Must be a vector or matrix.');
end

if any(any(~isfinite(Series)))
	error('econ:hpfilter:InvalidInputData', ...
		'Cannot have infinite or missing values in data.');
end

if NumSamples < 3		% treat samples with < 3 observations as trend data only
	warning('econ:hpfilter:InsufficientData', ...
		'Need at least three observations. Will just return input as Trend.');
	Trend = Series;
	Cyclical = zeros(NumSamples, NumSeries);
	return
end

if nargin < 2 || isempty(Smoothing)
	warning('econ:hpfilter:DefaultQuarterlySmoothing', ...
		'Missing or empty Smoothing parameter set to 1600 (quarterly data).');
	Smoothing = 1600;
end

if ~isvector(Smoothing) || ~isa(Smoothing,'double')
	error('econ:hpfilter:InvalidInputArg', ...
 		'Invalid format for Smoothing parameter. Must be a scalar or vector.');
else
	if ~any(numel(Smoothing) == [1, NumSeries])
		error('econ:hpfilter:InconsistentSmoothingDimensions', ...
			'Smoothing parameter is neither a scalar nor a conformable vector.');
	end
end

if any(isnan(Smoothing))
	error('econ:hpfilter:InvalidSmoothing', ...
		'Must have finite or infinite smoothing parameter.');
end

if any(Smoothing < 0)
	warning('econ:hpfilter:NegativeSmoothing', ...
		'Negative value for smoothing parameter. Will use absolute value.');
	Smoothing = abs(Smoothing);
end

% Step 2 - run the filter with either scalar or vector Smoothing

if (numel(Smoothing) == 1) || (max(Smoothing) == min(Smoothing))	% scalar
	if numel(Smoothing) > 1
		Smoothing = Smoothing(1);
	end
	if isinf(Smoothing)			% do OLS detrending
		Trend = Series - detrend(Series);
	else
		if NumSamples == 3			% special case with 3 samples
			A = eye(NumSamples, NumSamples) + ...
				Smoothing*[ 1 -2 1; -2 4 -2; 1 -2 1 ];
		else						% general case with > 3 samples
			e = repmat([Smoothing, -4*Smoothing, (1 + 6*Smoothing), ...
				-4*Smoothing, Smoothing], NumSamples, 1);
			A = spdiags(e, -2:2, NumSamples, NumSamples);
			A(1,1) = 1 + Smoothing;
			A(1,2) = -2*Smoothing;
			A(2,1) = -2*Smoothing;
			A(2,2) = 1 + 5*Smoothing;
			A(NumSamples - 1, NumSamples - 1) = 1 + 5*Smoothing;
			A(NumSamples - 1, NumSamples) = -2*Smoothing;
			A(NumSamples, NumSamples - 1) = -2*Smoothing;
			A(NumSamples, NumSamples) = 1 + Smoothing;
		end
		Trend = A \ Series;
	end
else																% vector
	Trend = zeros(NumSamples,NumSeries);
	if NumSamples == 3						% special case with 3 samples
		for i = 1:NumSeries
			if isinf(Smoothing(i))			% do OLS detrending
				Trend(:, i) = Series(:, i) - detrend(Series(:, i));
			else
				A = eye(NumSamples, NumSamples) + ...
					Smoothing(i)*[ 1 -2 1; -2 4 -2; 1 -2 1 ];
				Trend(:, i) = A \ Series(:, i);
			end
		end
	else									% general case with > 3 samples
		for i = 1:NumSeries
			if isinf(Smoothing(i))			% do OLS detrending
				Trend(:, i) = Series(:, i) - detrend(Series(:, i));
			else
				e = repmat([Smoothing(i), -4*Smoothing(i), (1 + 6*Smoothing(i)), ...
					-4*Smoothing(i), Smoothing(i)], NumSamples, 1);
				A = spdiags(e, -2:2, NumSamples, NumSamples);
				A(1,1) = 1 + Smoothing(i);
				A(1,2) = -2*Smoothing(i);
				A(2,1) = -2*Smoothing(i);
				A(2,2) = 1 + 5*Smoothing(i);
				A(NumSamples - 1, NumSamples - 1) = 1 + 5*Smoothing(i);
				A(NumSamples - 1, NumSamples) = -2*Smoothing(i);
				A(NumSamples, NumSamples - 1) = -2*Smoothing(i);
				A(NumSamples, NumSamples) = 1 + Smoothing(i);
				Trend(:, i) = A \ Series(:, i);
			end
		end
	end
end

% Step 3 - plot results if no output arguments

if nargout == 0
	figure(gcf);
	plot(Series,'b');
	hold on
	plot(Trend,'r');
	title('\bfHodrick-Prescott Filter');
	if NumSeries == 1
		legend('Raw Data','Smoothed Trend');
	end
	hold off;
elseif nargout > 1
	Cyclical = Series - Trend;
end
