function [beamformed_Image] = PCIfreqRBC(RF_Arr, element_Pos_Array_um_X, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um,p,range_frq,check)
% Performs PCI beamforming using a CUDA MEX function with precomputed delays.

    % Get dimensions
    [nfft, ncols] = size(RF_Arr); % nfft=Axial_depth, ncols=number of elements
    numX = length(image_Range_X_um); % Number of rows in beamformed image
    numZ = length(image_Range_Z_um); % Number of columns in beamformed image - FIXED THIS LINE
    total_pixels = numX * numZ;

    % --- Precompute all delays on the host ---
    disp('Precomputing all delays on host...');
    tic;

    all_delays_matrix = zeros(ncols, total_pixels); % Preallocate (ncols x total_pixels)
    pixel_index = 0; % Linear pixel index (1-based, column-major)

    % Loop Z first, then X to match MATLAB's column-major linearization
    for zi = 1:numZ
        currentZ_um = image_Range_Z_um(zi);
        for xi = 1:numX
            pixel_index = pixel_index + 1;
            currentX_um = image_Range_X_um(xi);

            % Calculate distance from pixel (xi, zi) to each element's position
            % element_Pos_Array_um is expected to be 2 x ncols [X_coords; Z_coords]
            dist_um = sqrt( (currentX_um - element_Pos_Array_um_X(1,:)).^2 + ...
                            (currentZ_um - element_Pos_Array_um_X(2,:)).^2 ); % Result is 1 x ncols

            % Calculate time-of-flight
            time_s = dist_um / speed_Of_Sound_umps; % Result is 1 x ncols

            % Calculate delay in samples (negative for focusing/DAS)
            delay_samples = -time_s * sampling_Freq; % Result is 1 x ncols

            % Store the delay vector for this pixel as a column
            all_delays_matrix(:, pixel_index) = delay_samples'; % Transpose to ncols x 1 before assignment
        end
    end
    precomputation_time = toc;
    disp(['Finished precomputing delays. Matrix size: ' num2str(size(all_delays_matrix)) '. Time: ' num2str(precomputation_time) 's']);
    % --- End Delay Precomputation ---

    % --- Call CUDA MEX function ---
    % Assumes the compiled MEX file from PCIimagingSparseCUDA_v6.cu is named 'withforcuda'
    disp('Calling CUDA MEX function (withforcuda)...');
    % RF_Arr = single(RF_Arr);
    % all_delays_matrix = single(all_delays_matrix);
    tic;
%    beamformed_Image = cudavsingle( ...
%     RF_Arr, ...                    % [nfft x ncols] real double
%     element_Pos_Array_um_X, ...    % [2 x ncols] real double (not used, but required for compatibility)
%     speed_Of_Sound_umps, ...       % scalar double (not used, but required for compatibility)
%     RF_Start_Time, ...             % scalar double (not used, but required for compatibility)
%     sampling_Freq, ...             % scalar double
%     image_Range_X_um, ...          % [numX x 1] real double
%     image_Range_Z_um, ...          % [numZ x 1] real double
%     p, ...                         % scalar double (p-th root)
%     all_delays_matrix, ...         % [ncols x (numX*numZ)] real double, precomputed delays (see below)
%     range_frq, ...                 % [1x2] double, frequency range [f_low, f_high]
%     check ...                      % scalar double, 1 for pCF, 0 for pDAS
% );

      beamformed_Image = cudavsingle( ...
    single(RF_Arr), ...                    % [nfft x ncols] real double
    single(element_Pos_Array_um_X), ...    % [2 x ncols] real double (not used, but required for compatibility)
    single(speed_Of_Sound_umps), ...       % scalar double (not used, but required for compatibility)
    single(RF_Start_Time), ...             % scalar double (not used, but required for compatibility)
    single(sampling_Freq), ...             % scalar double
    single(image_Range_X_um), ...          % [numX x 1] real double
    single(image_Range_Z_um), ...          % [numZ x 1] real double
    single(p), ...                         % scalar double (p-th root)
    single(all_delays_matrix), ...         % [ncols x (numX*numZ)] real double, precomputed delays (see below)
    single(range_frq), ...                 % [1x2] double, frequency range [f_low, f_high]
    single(check) ...                      % scalar double, 1 for pCF, 0 for pDAS
);
    cuda_execution_time = toc;
    beamformed_Image=double(beamformed_Image);
    disp(size(beamformed_Image))
    disp(['CUDA processing complete. Time: ' num2str(cuda_execution_time) 's']);
    % beamformed_Image is expected to be numX x numZ

end % End of function PCIimagingSparseupdated