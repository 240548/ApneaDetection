function class = apneaDetection(data)
    fs = 256;
    ecg = -double(data.EKG(:));
    ppg = -double(data.PPG(:));

    %% 1. ECG preprocessing & R-peak detection
    % bandpass 5-20 Hz
    [b_ecg, a_ecg] = butter(3, [5 20]/(fs/2), 'bandpass');
    ecg_filt = filtfilt(b_ecg, a_ecg, ecg);
    
    % Pan-Tompkins detector
    ecg_sq = ecg_filt.^2;
    ecg_integ = movmean(ecg_sq, round(0.1 * fs));
    min_peak_dist = round(0.6 * fs);
    [~, qrs_locs] = findpeaks(ecg_integ, 'MinPeakHeight', mean(ecg_integ), ...
                                         'MinPeakDistance', min_peak_dist);

    % local maximum search
    r_locs = zeros(size(qrs_locs));
    search_window = round(0.05 * fs); % 50ms window
    valid_count = 0;
    for i = 1:length(qrs_locs)
        curr_loc = qrs_locs(i);

        % window limits
        start_idx = max(1, curr_loc - search_window);
        end_idx = min(length(ecg_filt), curr_loc + search_window);

        % max amplitude in the raw filtered signal
        [~, max_rel_idx] = max(ecg_filt(start_idx:end_idx));

        % correct the location
        true_r_loc = start_idx + max_rel_idx - 1;

        valid_count = valid_count + 1;
        r_locs(valid_count) = true_r_loc;
    end

    % trim unused zeros if any
    r_locs = r_locs(1:valid_count);

    %% 2. ECG features
    if length(r_locs) < 5
        % signal likely noise or flatline
        rr_cv = 0; 
        edr_std = 0;
    else
        % Feature 1: HRV
        rr_raw = diff(r_locs) / fs;
        rr = rr_raw(rr_raw > 0.3 & rr_raw < 3.0); % remove physiological impossible values (0.3s to 2.0s)

        if ~isempty(rr)
             rr_cv = std(rr) / mean(rr); % coefficient of variation of RR (normalized HRV)
        else
             rr_cv = 0;
        end
  
        % Feature 2: EDR
        r_amps = ecg_filt(r_locs);
        if mean(r_amps) ~= 0
            r_amps = r_amps / mean(r_amps);
            edr_std = std(r_amps);
        else
            edr_std = 0;
        end 
    end

    %% 3. PPG preprocessing & P-peaks detection
    % bandpass 0.5-5 Hz
    [b_ppg, a_ppg] = butter(3, [0.5 5]/(fs/2), 'bandpass');
    ppg_filt = filtfilt(b_ppg, a_ppg, ppg);

    % systolic peaks detection
    [p_amps, p_locs] = findpeaks(ppg_filt, 'MinPeakDistance', min_peak_dist);

    %% 4. PPG features
    if length(p_locs) < 5
        % signal likely noise or flatline
        pp_cv = 0;
        ppg_amp_var = 0;
    else
        % Feature 3: PRV (PPG interval variability)
        pp_raw = diff(p_locs) / fs;
        pp = pp_raw(pp_raw > 0.3 & pp_raw < 2.0);
        if ~isempty(pp)
            pp_cv = std(pp) / mean(pp);
        else
            pp_cv = 0;
        end

        % Feature 4: PPG amplitude variability
        if mean(p_amps) ~= 0
            norm_p_amps = p_amps / mean(p_amps);
            ppg_amp_var = std(norm_p_amps);
        else
            ppg_amp_var = 0;
        end
    end

    % Feature 5: PPG baseline wander
    % calculate std to check for flatline
    ppg_sigma = std(ppg);
    
    % if signal is flat (std is near 0), skip processing
    if ppg_sigma < 1e-6
        baseline_std = 0;
    else
        % normalization
        ppg_z = (ppg - mean(ppg)) / ppg_sigma;
        
        % lowpass filter (< 0.5 Hz)
        [b_low, a_low] = butter(3, 0.5/(fs/2), 'low');
        ppg_baseline = filtfilt(b_low, a_low, ppg_z);
        
        % calculate wander
        baseline_std = std(ppg_baseline);
    end

    %% 4. Classification logic
   
    % Feature 1: HRV (heart rate variability)
    score_hrv = rr_cv > 0.19;

    % Feature 2: EDR variability
    score_edr = edr_std > 0.051;

    % Feature 3: PPG amplitude variability
    score_ppg = ppg_amp_var > 0.11;

    % Feature 4: PRV (pulse rate variability)
    score_prv = pp_cv > 0.2;

    % Feature 5: PPG baseline wander
    score_base = baseline_std > 0.5;

    % VOTING SYSTEM
    % we require at least 2 positive features
    votes = score_hrv + score_edr + score_ppg + score_prv + score_base;
    apnea = votes >= 2;

    class = double(apnea);
end
