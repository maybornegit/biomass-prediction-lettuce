function [fitresult, gof] = createFit(validationBiomass, predictedBiomass)
%CREATEFIT1(VALIDATIONBIOMASS,PREDICTEDBIOMASS)
%  Create a fit.
%
%  Data for 'Biomass' fit:
%      X Input : validationBiomass
%      Y Output: predictedbiomass
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 09-Apr-2018 14:12:41 自动生成


%% Fit: 'Biomass'.
[xData, yData] = prepareCurveData( validationBiomass, predictedBiomass );

% Set up fittype and options.
ft = fittype( 'poly1' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Robust = 'LAR';

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );

% Create a figure for the plots.
figure( 'Name', 'Biomass' );

% Plot fit with data.
% subplot( 2, 1, 1 );
h = plot( fitresult,'black', xData, yData,'*');
legend( h, 'predictedBiomass vs. measuredBiomass', 'Biomass', 'Location', 'NorthEast' );
% Label axes
xlabel measuredBiomass
ylabel predictedBiomass
grid on

% % Plot residuals.
% subplot( 2, 1, 2 );
% h = plot( fitresult, 'black', xData, yData, '*','residuals' );
% legend( h, 'Biomass - residuals', 'Zero Line', 'Location', 'NorthEast' );
% % Label axes
% xlabel measuredBiomass
% ylabel predictedBiomass
% grid on


