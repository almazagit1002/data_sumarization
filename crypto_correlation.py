import pandas as pd
import os
import logging
from typing import Dict


class CorrelationAnalyzer:
    """
    A class for computing and saving various types of correlations between cryptocurrencies and financial sectors.
    """
    
    def __init__(self, data_dir: str, output_dir: str = "correlation_outputs"):
        """
        Initialize the CorrelationAnalyzer.
        
        Args:
            data_dir (str): Directory containing the input CSV files
            output_dir (str): Directory to save correlation outputs
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir}")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('correlation_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("CorrelationAnalyzer initialized")

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a dataset with date as index.
        
        Args:
            filename (str): Name of the CSV file to load
            
        Returns:
            pd.DataFrame: Loaded dataframe with date index
        """
        filepath = os.path.join(self.data_dir, filename)
        self.logger.info(f"Loading dataset: {filepath}")
        
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"Successfully loaded {filename} with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {str(e)}")
            raise

    def compute_and_save_correlation(self, crypto_df: pd.DataFrame, sector_df: pd.DataFrame, sector_name: str):
        """
        Compute and save static correlations between cryptocurrencies and sector data.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
            sector_df (pd.DataFrame): Sector data
            sector_name (str): Name of the sector for output file naming
        """
        self.logger.info(f"Computing static correlation for {sector_name}")
        
        # Align by date, inner join to keep only common dates
        combined = crypto_df.join(sector_df, how='inner')
        self.logger.debug(f"Combined dataset shape after join: {combined.shape}")
        
        output_file = os.path.join(self.output_dir, f"correlation_{sector_name}_vs_crypto.txt")
        
        try:
            with open(output_file, "w") as f:
                f.write(f"Correlation between Cryptos and {sector_name} dataset\n\n")
                
                # For each crypto column
                for crypto_col in crypto_df.columns:
                    f.write(f"Correlations for crypto: {crypto_col}\n")
                    # Calculate correlation of this crypto with each sector column
                    correlations = combined.corr()[crypto_col][sector_df.columns]
                    
                    for sector_col, corr_value in correlations.items():
                        f.write(f"  {sector_col}: {corr_value:.4f}\n")
                    f.write("\n")
            
            self.logger.info(f"Static correlation saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save static correlation for {sector_name}: {str(e)}")
            raise

    def compute_and_save_rolling_correlation(self, crypto_df: pd.DataFrame, sector_df: pd.DataFrame, 
                                           sector_name: str, window: int = 30):
        """
        Compute and save rolling correlations between cryptocurrencies and sector data.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
            sector_df (pd.DataFrame): Sector data
            sector_name (str): Name of the sector for output file naming
            window (int): Rolling window size in days
        """
        self.logger.info(f"Computing rolling correlation for {sector_name} with window={window}")
        
        # Align on dates - inner join
        combined = crypto_df.join(sector_df, how='inner')
        
        # Create output folder if not exists
        sector_output_dir = os.path.join(self.output_dir, f"rolling_correlation_{sector_name}")
        os.makedirs(sector_output_dir, exist_ok=True)
        self.logger.debug(f"Created rolling correlation directory: {sector_output_dir}")
        
        # For each crypto column and each sector column, compute rolling correlation
        for crypto_col in crypto_df.columns:
            for sector_col in sector_df.columns:
                try:
                    rolling_corr = combined[crypto_col].rolling(window).corr(combined[sector_col])
                    
                    # Save rolling correlation series to CSV
                    output_file = os.path.join(sector_output_dir, f"{crypto_col}_vs_{sector_col}.csv")
                    rolling_corr.index.name = "Date"
                    rolling_corr.to_csv(output_file, header=["rolling_correlation"])
                    
                except Exception as e:
                    self.logger.error(f"Failed to compute rolling correlation for {crypto_col} vs {sector_col}: {str(e)}")
                    continue
        
        self.logger.info(f"Rolling correlations saved to: {sector_output_dir}")

    def compute_and_save_lagged_correlation(self, crypto_df: pd.DataFrame, sector_df: pd.DataFrame, 
                                          sector_name: str, max_lag: int = 5):
        """
        Compute and save lagged correlations between sector data and cryptocurrencies.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
            sector_df (pd.DataFrame): Sector data
            sector_name (str): Name of the sector for output file naming
            max_lag (int): Maximum number of lag days to compute
        """
        self.logger.info(f"Computing lagged correlation for {sector_name} with max_lag={max_lag}")
        
        combined = crypto_df.join(sector_df, how='inner')
        output_file = os.path.join(self.output_dir, f"lagged_correlation_{sector_name}_vs_crypto.txt")
        
        try:
            with open(output_file, "w") as f:
                f.write(f"Lagged Correlation between {sector_name} and Crypto (lags from 1 to {max_lag} days)\n\n")
                
                for lag in range(1, max_lag + 1):
                    f.write(f"=== Lag: {lag} days (sector leads crypto) ===\n")
                    shifted_sector_df = sector_df.shift(lag)
                    combined_shifted = crypto_df.join(shifted_sector_df, how='inner')
                    
                    for crypto_col in crypto_df.columns:
                        f.write(f"Crypto: {crypto_col}\n")
                        correlations = combined_shifted.corr()[crypto_col][sector_df.columns]
                        
                        for sector_col, corr_value in correlations.items():
                            f.write(f"  {sector_col} (lag {lag}): {corr_value:.4f}\n")
                        f.write("\n")
            
            self.logger.info(f"Lagged correlation saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save lagged correlation for {sector_name}: {str(e)}")
            raise

    def compute_and_save_crypto_static_correlation(self, crypto_df: pd.DataFrame):
        """
        Compute and save static correlations between cryptocurrencies.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
        """
        self.logger.info("Computing static correlation between cryptocurrencies")
        
        output_file = os.path.join(self.output_dir, "crypto_static_correlation.txt")
        correlation_matrix = crypto_df.corr()

        try:
            with open(output_file, "w") as f:
                f.write("Static Correlation Matrix Between Cryptocurrencies\n\n")
                f.write(correlation_matrix.to_string(float_format=lambda x: f"{x:.4f}"))
            
            self.logger.info(f"Crypto static correlation saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save crypto static correlation: {str(e)}")
            raise

    def compute_and_save_crypto_rolling_correlation(self, crypto_df: pd.DataFrame, window: int = 30):
        """
        Compute and save rolling correlations between cryptocurrencies.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
            window (int): Rolling window size in days
        """
        self.logger.info(f"Computing rolling correlation between cryptocurrencies with window={window}")
        
        output_dir = os.path.join(self.output_dir, "crypto_rolling_correlation")
        os.makedirs(output_dir, exist_ok=True)
        self.logger.debug(f"Created crypto rolling correlation directory: {output_dir}")
        
        for i, crypto1 in enumerate(crypto_df.columns):
            for crypto2 in crypto_df.columns[i+1:]:
                try:
                    rolling_corr = crypto_df[crypto1].rolling(window).corr(crypto_df[crypto2])
                    rolling_corr.index.name = "Date"
                    output_file = os.path.join(output_dir, f"{crypto1}_vs_{crypto2}_rolling.csv")
                    rolling_corr.to_csv(output_file, header=["rolling_correlation"])
                except Exception as e:
                    self.logger.error(f"Failed to compute rolling correlation for {crypto1} vs {crypto2}: {str(e)}")
                    continue
        
        self.logger.info(f"Crypto rolling correlations saved to: {output_dir}")

    def compute_and_save_crypto_lagged_correlation(self, crypto_df: pd.DataFrame, max_lag: int = 1):
        """
        Compute and save lagged correlations between cryptocurrencies.
        
        Args:
            crypto_df (pd.DataFrame): Cryptocurrency data
            max_lag (int): Maximum number of lag days to compute
        """
        self.logger.info(f"Computing lagged correlation between cryptocurrencies with max_lag={max_lag}")
        
        output_file = os.path.join(self.output_dir, "crypto_lagged_correlation.txt")
        
        try:
            with open(output_file, "w") as f:
                f.write(f"Lagged Correlations Between Cryptocurrencies (lags 1 to {max_lag})\n\n")
                
                for lag in range(1, max_lag + 1):
                    f.write(f"=== Lag: {lag} days (crypto1 leads crypto2) ===\n")
                    shifted_df = crypto_df.shift(lag)
                    combined = shifted_df.join(crypto_df, how='inner', lsuffix='_lead', rsuffix='_lag')

                    for crypto1 in crypto_df.columns:
                        for crypto2 in crypto_df.columns:
                            if crypto1 == crypto2:
                                continue
                            corr = combined[f"{crypto1}_lead"].corr(combined[f"{crypto2}_lag"])
                            f.write(f"{crypto1} (lag {lag}) -> {crypto2}: {corr:.4f}\n")
                        f.write("\n")
            
            self.logger.info(f"Crypto lagged correlation saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save crypto lagged correlation: {str(e)}")
            raise

    def run_analysis(self):
        """
        Run the complete correlation analysis for all cryptocurrencies and sectors.
        """
        self.logger.info("Starting complete correlation analysis")
        
        try:
            # Load cryptocurrency data
            crypto_df = self.load_dataset("cryptos.csv")
            
            # Compute crypto-to-crypto correlations
            self.compute_and_save_crypto_static_correlation(crypto_df)
            self.compute_and_save_crypto_rolling_correlation(crypto_df, window=30)
            self.compute_and_save_crypto_lagged_correlation(crypto_df, max_lag=1)

            # Define sector files
            sector_files = {
                "key_sectors": "key_sectors.csv",
                "financial": "financial.csv",
                "stocks": "stocks.csv",
                "commodities": "commodities.csv",
                "bonds_interest": "bonds_interest.csv"
            }

            # Process each sector
            for sector_name, filename in sector_files.items():
                self.logger.info(f"Processing sector: {sector_name}")
                try:
                    sector_df = self.load_dataset(filename)
                    self.compute_and_save_correlation(crypto_df, sector_df, sector_name)
                    self.compute_and_save_rolling_correlation(crypto_df, sector_df, sector_name, window=30)
                    self.compute_and_save_lagged_correlation(crypto_df, sector_df, sector_name, max_lag=1)
                    self.logger.info(f"Completed analysis for sector: {sector_name}")
                except Exception as e:
                    self.logger.error(f"Failed to process sector {sector_name}: {str(e)}")
                    continue
            
            self.logger.info("Complete correlation analysis finished successfully")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise


def main():
    """Main function to run the correlation analysis."""
    DATA_DIR = r"C:\Users\Maza\Desktop\USD-strenght-analysis\data"
    OUTPUT_DIR = r"correlation_outputs"
    
    analyzer = CorrelationAnalyzer(DATA_DIR, OUTPUT_DIR)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()