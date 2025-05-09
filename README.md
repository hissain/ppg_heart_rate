# Heart Rate and Respiration Rate Detection from Dummy PPG Sensor

This project demonstrates the generation, processing, and analysis of synthetic Photoplethysmogram (PPG) data to detect heart rate and respiration rate. The implementation includes generating synthetic PPG data, saving it to a file, analyzing it using a custom `PPGRespirationAnalyzer` class, and visualizing the results.

![Example](output/ppg_analysis_results.png)

## Features

- **Synthetic PPG Data Generation**: Simulates PPG signals with components for heart rate, respiration, baseline drift, and noise.
- **Data Saving**: Saves the generated PPG data and timestamps to a JSON file.
- **PPG Analysis**: Detects heart rate and respiration rate from the PPG signal using preprocessing and signal analysis techniques.
- **Visualization**: Plots the results of the analysis and saves the output as an image.

## Requirements

The project requires the following Python libraries, which can be installed using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Dependencies:
- **numpy:** For numerical computations.
- **matplotlib:** For plotting and visualization.
- **scipy:** For signal processing.
- **pandas:** For handling data files.


## File Descriptions
### sample.py

- **generate_sample_data:** Generates synthetic PPG data with configurable duration and sampling rate.
- **save_sample_data:** Saves the generated PPG data and timestamps to a JSON file.
- **main:** Orchestrates the data generation, saving, analysis, and visualization.

### ppg.py

**PPGRespirationAnalyzer:** A class for analyzing PPG data.

**_load_data:_** Loads PPG data from JSON or CSV files.
**_preprocess_signal:_** Preprocesses the PPG signal by removing noise, filtering, and detrending.

**_analyze_ppg:_** (Not fully shown) Analyzes the PPG signal to extract heart rate and respiration rate.

### requirements.txt

Lists the required Python libraries for the project.

## Usage

- **Generate and Save PPG Data:** Run the sample.py script to generate synthetic PPG data and save it to a file:
- **Analyze PPG Data:** The script will analyze the generated data using the PPGRespirationAnalyzer class and display the results, including:
* Heart rate (in beats per minute),
* Respiration rate (in breaths per minute)

-**Visualize Results:** The analysis results will be plotted and saved as an image in the output directory.

## Output

- **Generated Data:** Saved as output/sample_ppg_data.json.
- **Analysis Results:** Saved as output/ppg_analysis_results.png.

## Example Results
Below is an example of the results obtained from analyzing synthetic PPG data:

```
Heart Rate: 72 bpm
Respiration Rate: 15 breaths/min
```

## Notes
The synthetic data generation simulates realistic PPG signals but may not fully represent real-world data.
Ensure the output directory exists before running the script to avoid file-saving errors.

## License
This project is for demonstration purposes and is provided under an open-source license.

