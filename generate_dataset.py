import numpy as np
import h5py
import pycbc.noise
import pycbc.psd
import pycbc.detector
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigmasq

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
PARAMS = {
    'seed': 42,                  
    'num_samples': 100,          
    'output_file': 'train_data.hdf5',
    
    'mass_range': [30, 80],
    'spin_range': [0, 0.998],
    'snr_range':  [10, 50],
    
    'duration': 32.0,
    'sample_rate': 2048,
    'f_lower': 20.0,
    'approximant': 'IMRPhenomPv2',
    
    'slice_start': -0.20,
    'slice_end':   +0.05
}

PSD_FILES = {
    'H1': 'sensitivities/aligo_O4high.txt',
    'L1': 'sensitivities/aligo_O4high.txt',
    'V1': 'sensitivities/avirgo_O4high_NEW.txt',
    'K1': 'sensitivities/kagra_128Mpc.txt'
}

# ==========================================
# 2. 関数定義
# ==========================================
def sample_parameters():
    m1 = np.random.uniform(*PARAMS['mass_range'])
    m2 = np.random.uniform(*PARAMS['mass_range'])
    mass1, mass2 = max(m1, m2), min(m1, m2)

    spin1z = np.random.uniform(*PARAMS['spin_range'])
    spin2z = np.random.uniform(*PARAMS['spin_range'])

    ra = np.random.uniform(0, 2 * np.pi)
    dec = np.arcsin(np.random.uniform(-1, 1))
    pol = np.random.uniform(0, 2 * np.pi)
    inc = np.arccos(np.random.uniform(-1, 1))
    coa_phase = np.random.uniform(0, 2 * np.pi)
    target_snr = np.random.uniform(*PARAMS['snr_range'])

    return {
        'mass1': mass1, 'mass2': mass2, 'spin1z': spin1z, 'spin2z': spin2z,
        'ra': ra, 'dec': dec, 'inclination': inc, 'polarization': pol,
        'coa_phase': coa_phase, 'target_snr': target_snr
    }

def load_psd(det_name, flen, delta_f):
    path = PSD_FILES.get(det_name)
    try:
        psd = pycbc.psd.from_txt(path, flen, delta_f, PARAMS['f_lower'], is_asd_file=True)
    except:
        psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, PARAMS['f_lower'])
        if det_name == 'K1': psd.data *= 10.0
    return psd

# ==========================================
# 3. メイン生成ループ
# ==========================================
def generate_dataset():
    np.random.seed(PARAMS['seed'])
    
    num_samples = PARAMS['num_samples']
    sample_rate = PARAMS['sample_rate']
    slice_points = int(round((PARAMS['slice_end'] - PARAMS['slice_start']) * sample_rate))

    print(f"Generating {num_samples} samples (Seed: {PARAMS['seed']})...")

    data_buffer = {
        'h1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'l1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'v1_strain': np.zeros((num_samples, slice_points), dtype='float32'),
        'k1_strain': np.zeros((num_samples, slice_points), dtype='float32')
    }
    
    param_buffer = {k: np.zeros(num_samples) for k in 
                    ['mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec', 'injection_snr']}

    detectors = {name: pycbc.detector.Detector(name) for name in ['H1', 'L1', 'V1', 'K1']}
    det_indices = {'H1': 0, 'L1': 1, 'V1': 2, 'K1': 3}

    for i in range(num_samples):
        p = sample_parameters()
        
        # 1. 基準波形の生成 (合体時刻が必ず絶対時間の t=0.0 になる)
        hp, hc = get_td_waveform(
            approximant=PARAMS['approximant'],
            mass1=p['mass1'], mass2=p['mass2'],
            spin1z=p['spin1z'], spin2z=p['spin2z'],
            distance=1000.0,
            inclination=p['inclination'],
            coa_phase=p['coa_phase'],
            delta_t=1.0/sample_rate,
            f_lower=PARAMS['f_lower']
        )

        net_snr_sq = 0.0
        signals_no_noise = {}

        # 2. 投影とSNR計算
        for name, det in detectors.items():
            fp, fc = det.antenna_pattern(p['ra'], p['dec'], p['polarization'], 0.0)
            dt = det.time_delay_from_earth_center(p['ra'], p['dec'], 0.0)
            
            sig = fp * hp + fc * hc
            sig.start_time += dt # 空間的な到達時間差を正確に反映
            
            # SNR計算のため波形を32秒にパディング
            sig_snr = sig.copy()
            sig_snr.resize(int(PARAMS['duration'] * sample_rate))
            flen = int(len(sig_snr) / 2) + 1
            psd = load_psd(name, flen, sig_snr.delta_f)
            
            snr_sq = sigmasq(sig_snr, psd=psd, low_frequency_cutoff=PARAMS['f_lower'])
            net_snr_sq += snr_sq
            
            signals_no_noise[name] = (sig, psd)

        current_net_snr = np.sqrt(net_snr_sq)
        if current_net_snr == 0: current_net_snr = 1e-10
        scaling_factor = current_net_snr / p['target_snr']

        # 3. ノイズへの波形の埋め込みと、厳密な切り出し
        for name, (sig, psd) in signals_no_noise.items():
            sig_final = sig / scaling_factor
            
            # 32秒間のノイズを生成 (t=0が中央になるよう -16.0秒開始に設定)
            noise_seed = PARAMS['seed'] + (i * 100) + det_indices[name]
            noise_len = int(PARAMS['duration'] * sample_rate)
            noise = pycbc.noise.noise_from_psd(noise_len, sig_final.delta_t, psd, seed=noise_seed)
            noise.start_time = -(PARAMS['duration'] / 2.0) 
            
            strain_data = noise.numpy().copy()
            
            # 信号(sig_final)を、絶対時間を基準にしてノイズ配列上の正しいインデックスへ加算
            start_idx = int(round((float(sig_final.start_time) - float(noise.start_time)) * sample_rate))
            end_idx = start_idx + len(sig_final)
            
            slice_start, slice_end = max(0, start_idx), min(len(strain_data), end_idx)
            sig_start = max(0, -start_idx)
            sig_end = sig_start + (slice_end - slice_start)
            
            if slice_end > slice_start:
                strain_data[slice_start:slice_end] += sig_final.numpy()[sig_start:sig_end]
                
            # 4. 指定範囲 (-0.20秒 〜 +0.05秒) の抽出
            ext_start_time = PARAMS['slice_start']
            ext_start_idx = int(round((float(ext_start_time) - float(noise.start_time)) * sample_rate))
            ext_end_idx = ext_start_idx + slice_points
            
            data_slice = strain_data[ext_start_idx:ext_end_idx]
            
            if len(data_slice) < slice_points:
                data_slice = np.pad(data_slice, (0, slice_points - len(data_slice)))
                
            key_name = name.lower() + '_strain'
            data_buffer[key_name][i] = data_slice

        # パラメータ保存
        for k in ['mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec']:
            param_buffer[k][i] = p[k]
        param_buffer['injection_snr'][i] = p['target_snr']

        if (i+1) % 100 == 0 or (i+1) == num_samples:
            print(f"Processed {i+1}/{num_samples} samples.")

    print("Saving to HDF5 (ggwd structure)...")
    with h5py.File(PARAMS['output_file'], 'w') as f:
        grp_samples = f.create_group('injection_samples')
        grp_params = f.create_group('injection_parameters')
        for key, val in data_buffer.items(): grp_samples.create_dataset(key, data=val)
        for key, val in param_buffer.items(): grp_params.create_dataset(key, data=val)
    print(f"Done. Saved to {PARAMS['output_file']}")

if __name__ == "__main__":
    generate_dataset()