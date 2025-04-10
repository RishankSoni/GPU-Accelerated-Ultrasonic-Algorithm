function [beamformed_Image] = PCIimagingSparseupdated(RF_Arr, element_Pos_Array_um, speed_Of_Sound_umps, RF_Start_Time, sampling_Freq, image_Range_X_um, image_Range_Z_um, p)

    [nfft, ncols] = size(RF_Arr); 
    numX = length(image_Range_X_um); 
    numZ = length(image_Range_Z_um); 
    total_pixels = numX * numZ;
    %Precomputation of all the delay
    tic;
    all_delays_matrix = zeros(ncols, total_pixels); 
    pixel_index = 0; 
    for zi = 1:numZ
        currentZ_um = image_Range_Z_um(zi);
        for xi = 1:numX
            pixel_index = pixel_index + 1;
            currentX_um = image_Range_X_um(xi);
            dist_um = sqrt( (currentX_um - element_Pos_Array_um(1,:)).^2 + (currentZ_um - element_Pos_Array_um(2,:)).^2 ); 
            time_s = dist_um / speed_Of_Sound_umps; 
            delay_samples = -time_s * sampling_Freq; 

            all_delays_matrix(:, pixel_index) = delay_samples'; 
        end
    end
    precomputation_time = toc;
    disp(['Finished precomputing delays. Matrix size: ' num2str(size(all_delays_matrix)) '. Time: ' num2str(precomputation_time) 's']);
    tic;
    %Running the CUDA Mex Function
    [beamformed_Image] = PCI_cuda(RF_Arr, element_Pos_Array_um, speed_Of_Sound_umps,RF_Start_Time,sampling_Freq,image_Range_X_um,image_Range_Z_um,p, all_delays_matrix);                          
    cuda_execution_time = toc;
    disp(['CUDA processing complete Time: ' num2str(cuda_execution_time) 's']);
 

end 