"""
This module contains components for data preprocessing pipeline:

DataExplorer
    - Missing value analysis
    - Basic statistics
    - Data quality checks

DataCleaner
    - Basic data cleaning
    - Categorical validation
    - Vital signs cleaning
    - Blood pressure validation

FeatureEngineer
    - Time-based features
    - Interaction features
    - Domain-specific features

DataTransformer (to be implemented)
    - Scaling
    - Encoding
    - Dimensionality reduction

DataPruner (to be implemented)
    - Outlier removal
    - Feature selection
    - Sample balancing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import boxcox
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

__all__ = [
    'DataExplorer',
    'DataCleaner',
    'FeatureEngineer',
    'DataTransformer',
    'PCATransformer'
]

FEATURE_SETS = {
    'original_numerical': [
        'age_at_ed', 'temperature', 'heartrate', 'resprate',
        'o2sat', 'sbp', 'dbp', 'pain', 'acuity'
    ],
    'original_categorical': [
        'gender', 'arrival_transport', 'disposition', 'race', 'dow'
    ],
    'original_temporal': [
        'hour', 'minute'
    ],
    'derived_numerical': [
        'map', 'pulse_pressure', 'shock_index', 
        'rate_pressure_product', 'temp_delta',
        'week_minutes', 'day_minutes'
    ],
    'derived_binary': [
        'is_tachycardic', 'is_bradycardic', 'is_hypoxic',
        'is_hypertensive', 'is_hypotensive', 'is_febrile',
        'is_hypothermic', 'has_sirs', 'is_daytime'
    ],
    'derived_categorical': [
        'hr_category', 'resp_category', 'o2_category',
        'sbp_category', 'dbp_category', 'temp_category',
        'pain_category', 'age_group', 'acuity_group'
    ],
    'box_transformed': [
        'temperature_beta',
        'o2sat_ibx',
        'shock_index_bx',
        'rate_pressure_product_bx',
        'resprate_bx',
        'pain_bx'
    ],
    'cyclical_features': [
        'week_minutes_sin', 'week_minutes_cos',
        'day_minutes_sin', 'day_minutes_cos'
    ],
    'gmm_features': [
        'age_at_ed_ss',
        'temperature_beta',
        'temp_delta_ss',
        'heartrate_ss',
        'resprate_bx',
        'o2sat_ibx',
        'sbp_bx',
        'dbp_bx',
        'pain_bx',
        'map_bx',
        'pulse_pressure_bx',
        'shock_index_bx',
        'rate_pressure_product_bx',
        'week_minutes_sin',
        'week_minutes_cos',
        'day_minutes_sin',
        'day_minutes_cos'
    ],
    'metadata': [
        'subject_id', 'stay_id'
    ]
}

def get_features(data: pd.DataFrame, feature_sets: list) -> pd.DataFrame:
    """
    Retrieve a DataFrame with specified feature sets.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        feature_sets (list): List of feature set names from FEATURE_SETS
        
    Returns:
        pd.DataFrame: DataFrame with only the specified feature sets
    """
    selected_features = []
    for feature_set in feature_sets:
        selected_features.extend(FEATURE_SETS[feature_set])
    return data[selected_features]

class DataExplorer:
    """
    A class to perform exploratory data analysis operations.
    
    This class provides methods for initial data exploration, including
    analysis of missing values, distribution analysis, and data quality checks.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataExplorer with a dataset.
        
        Args:
            data (pd.DataFrame): The dataset to analyze
        """
        self.data = data.copy()
    
    def summarize_missing_data(self) -> pd.DataFrame:
        """
        Analyzes and visualizes missing data patterns in the dataset.
        
        Returns:
            pd.DataFrame: Summary DataFrame with columns:
                - Missing_Count: Number of missing values per column
                - Missing_Percentage: Percentage of missing values per column
                
        Note:
            Also generates a bar plot showing missing percentages for each variable
        """
        missing_summary = pd.DataFrame({
            'Missing_Count': self.data.isnull().sum(),
            'Missing_Percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
        }).sort_values('Missing_Percentage', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=missing_summary.index, 
                    y='Missing_Percentage',
                    data=missing_summary)
        plt.xticks(rotation=45, ha='right')
        plt.title('Missing Data Percentage by Variable')
        plt.ylabel('Missing Percentage')
        plt.tight_layout()
        plt.show()
        
        return missing_summary

    def analyze_missing_correlations(self) -> pd.DataFrame:
        """
        Analyzes and visualizes correlations between missing values in different features.
        
        Creates a correlation heatmap showing how missing values are related across
        features that have missing data. Useful for identifying patterns in missingness
        and potential MCAR/MAR/MNAR mechanisms.
        
        Returns:
            pd.DataFrame: Correlation matrix of missingness patterns
        """
        print("\nVisualization Caption:")
        print("Heatmap showing correlations between missing values across features.\n")
        
        # Exclude ID columns
        exclude_columns = ['subject_id', 'stay_id']
        features = [col for col in self.data.columns if col not in exclude_columns]
        
        # Create binary missingness indicators
        missing_binary = self.data[features].isnull().astype(int)
        
        # Compute correlation between missing indicators
        missing_corr = missing_binary.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(missing_corr, 
                    annot=True, 
                    cmap='YlOrRd',
                    fmt='.2f',
                    center=0,
                    annot_kws={'size': 8})
        plt.title('Correlation of Missingness Patterns\n(Features with Missing Values)')
        plt.tight_layout()
        plt.show()
        
        return missing_corr

    def _get_missing_counts(self) -> pd.DataFrame:
        """
        Prepares data for missing counts analysis.
        
        Returns:
            pd.DataFrame: DataFrame with missing indicators and grouping columns
        """
        # Define vital signs
        vital_signs = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
        
        # Create a copy of data to avoid modifications to original
        _df = self.data.copy()
        
        # Create missing indicators
        _df['missing_vitals'] = _df[vital_signs].isnull().any(axis=1).astype(int)
        _df['missing_age'] = _df['age_at_ed'].isnull().astype(int)
        _df['missing_pain'] = _df['pain'].isnull().astype(int)
        _df['missing_acuity'] = _df['acuity'].isnull().astype(int)
        
        # Keep only relevant columns
        _df = _df[['missing_vitals', 'missing_age', 'missing_pain', 'missing_acuity', 
                   'arrival_transport', 'disposition']].reset_index(drop=True)
        
        return _df

    def plot_missingness_distribution(self):
        """
        Creates a 2x3 grid of charts analyzing missing data patterns across arrival modes
        and dispositions, with annotations of missingness ratios.
        """
        def setup_figure():
            """Create and setup the figure and axes"""
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.1], width_ratios=[0.1, 1, 1, 1])
            axes = np.array([
                [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])],
                [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])],
            ])
            return fig, gs, axes

        def format_axes(axes):
            """Format all subplot axes"""
            for ax in axes.flat:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.xaxis.set_major_formatter(lambda x, p: f'{int(x/1000)}')
                ax.grid(True, axis='x', alpha=0.75)
                ax.set_xlim(0, 300000)

            # Format top row
            for ax in axes[0]:
                ax.xaxis.set_ticklabels([])
                ax.spines['bottom'].set_visible(False)

            # Format right columns
            for i in range(2):
                for j in range(1, 3):
                    axes[i,j].yaxis.set_visible(False)
                    axes[i,j].spines['left'].set_visible(False)

            # Add x-labels
            for ax in axes[1]:
                ax.set_xlabel('Count (Thousands)')

        def add_labels(fig, gs):
            """Add row and column labels"""
            # Row labels
            for idx, label in enumerate(['Arrival', 'Disposition']):
                ax = fig.add_subplot(gs[idx, 0])
                ax.text(0.5, 0.5, label, rotation=90, ha='center', va='center',
                       fontsize=18, fontweight='bold')
                ax.axis('off')

            # Column labels
            for idx, label in enumerate(['Missing Vitals', 'Missing Acuity', 'Missing Pain']):
                ax = fig.add_subplot(gs[2, idx+1])
                ax.text(0.5, 0.5, label, ha='center', va='center',
                       fontsize=18, fontweight='bold')
                ax.axis('off')

        def plot_missing_data(ax, group_data, missing_col, color):
            """Plot and annotate missing data bars for a given group"""
            counts = group_data.size()
            missing_counts = group_data[missing_col].sum()
            sort_idx = counts.sort_values(ascending=True).index
            counts = counts[sort_idx]
            missing_counts = missing_counts[sort_idx]
            
            y_pos = np.arange(len(counts))
            ax.barh(y_pos, counts, color='lightgray', alpha=0.5, label='Total Count')
            bars = ax.barh(y_pos, missing_counts, color=color, alpha=0.7, 
                          label=f'Missing {missing_col.replace("missing_", "")}')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([to_camel_case(str(x)) for x in sort_idx])

            # Annotate bars
            for i, bar in enumerate(bars):
                total_val = counts.iloc[i]
                missing_val = missing_counts.iloc[i]
                if total_val > 0:
                    ratio = int((missing_val / total_val) * 100)
                    ax.text(bar.get_width() + 5000,
                           bar.get_y() + bar.get_height() / 2,
                           f"{ratio}%",
                           va='center', ha='left',
                           color='black',
                           fontsize=11,
                           fontweight='bold')

        def to_camel_case(text):
            """Convert text to camel case"""
            words = text.replace('_', ' ').title().split()
            camel_case = words[0] + ' ' + ' '.join(word.capitalize() for word in words[1:])
            return camel_case if len(camel_case) <= 15 else camel_case[:11] + '...'

        # Main execution
        _df = self._get_missing_counts()
        fig, gs, axes = setup_figure()
        format_axes(axes)
        add_labels(fig, gs)

        # Plot all groups
        groups = [('arrival_transport', 0), ('disposition', 1)]
        missing_cols = [('missing_vitals', 'skyblue'), 
                       ('missing_acuity', 'lightcoral'), 
                       ('missing_pain', 'plum')]
        
        for group_col, row in groups:
            for (missing_col, color), col in zip(missing_cols, range(3)):
                plot_missing_data(axes[row,col], _df.groupby(group_col), missing_col, color)

        # Add titles
        fig.suptitle('Missing Data Analysis in Emergency Department Records', 
                     y=0.95, fontsize=22, weight='bold')
        plt.figtext(0.5, 0.91, 
                    'Distribution of missing vital signs, age, and pain data across arrival modes and dispositions',
                    ha='center', fontsize=18)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    @staticmethod
    def plot_features_distribution(data: pd.DataFrame) -> None:
        """
        Creates a grid of violin plots showing the distribution of key features.
        
        Args:
            data (pd.DataFrame): DataFrame containing the features to plot
            
        Note:
            Generates violin plots with embedded box plots and statistics for:
            - Demographic features (age)
            - Vital signs (temperature, heart rate, etc.)
            - Clinical scores (pain, acuity)
        """
        plt.style.use('default')  # Using default style instead of seaborn
        features = ['age_at_ed', 'temperature', 'heartrate', 'resprate', 
                   'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
        n_features = len(features)
        n_rows = (n_features + 2) // 3  # Calculate number of rows needed

        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.ravel()  # Flatten axes array for easier indexing

        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Create violin plot with boxplot inside
            sns.violinplot(data=data, y=feature, ax=ax, inner='box', color='lightblue')
            
            # Add title and labels
            ax.set_title(f'{feature} Distribution')
            ax.set_ylabel(feature)
            
            # Add text with basic statistics
            stats_text = f'Mean: {data[feature].mean():.2f}\n'
            stats_text += f'Median: {data[feature].median():.2f}\n'
            stats_text += f'Missing: {data[feature].isna().sum()/len(data)*100:.1f}%'
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Remove any empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_qq_plot(data: pd.DataFrame, features: list | str):
        """
        Plot QQ plots for given features.
        
        Args:
            data (pd.DataFrame): Input data
            features (list or str): Feature name(s) to plot
        """
        # Convert single feature to list
        if isinstance(features, str):
            features = [features]
        
        # Calculate number of rows needed
        n_features = len(features)
        n_cols = 3  # Use 3 columns
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.ravel()  # Flatten axes array
        
        # Create QQ plot for each feature
        for idx, feature in enumerate(features):
            stats.probplot(data[feature], dist='norm', plot=axes[idx])
            axes[idx].set_title(f'Q-Q Plot: {feature}')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.show()
        

class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        
        # Define valid values for categorical variables
        self.valid_transport = ['AMBULANCE', 'WALK IN']
        self.valid_disposition = ['HOME', 'ADMITTED']
        
        # Define expected ranges for vital signs
        self.vital_ranges = {
            'temperature': (56, 111.4),   # Fahrenheit
            'heartrate': (13, 250),       # beats per minute
            'resprate': (1, 97),          # breaths per minute
            'o2sat': (50, 100),            # percentage
            'sbp': (5, 299),              # mmHg
            'dbp': (0, 297),              # mmHg
            'pain': (0, 10),              # pain scale
            'acuity': (1, 5)              # acuity scale
        }

        self.data_clean = self.prepare_data()

    def prepare_data(self):
        """Main method to clean and prepare the data"""
        df = self.data.copy()
        
        # Clean categorical variables
        df = self._clean_categorical(df)
        
        # Clean vital signs
        df = self._clean_vitals(df)
        
        # Clean blood pressure (special case due to relationship between sbp/dbp)
        df = self._clean_blood_pressure(df)
        
        return df

    def _clean_categorical(self, df):
        """Clean categorical variables"""
        # Filter to valid transport modes
        df.loc[~df['arrival_transport'].isin(self.valid_transport), 'arrival_transport'] = np.nan
        
        # Filter to valid dispositions
        df.loc[~df['disposition'].isin(self.valid_disposition), 'disposition'] = np.nan
        
        return df

    def _clean_vitals(self, df):
        """Clean vital signs using physiological ranges"""
        for vital, (min_val, max_val) in self.vital_ranges.items():
            if vital not in ['sbp', 'dbp']:  # Blood pressure handled separately
                df.loc[~df[vital].between(min_val, max_val), vital] = np.nan
        return df

    def _clean_blood_pressure(self, df):
        """Clean blood pressure readings using statistical and physiological rules"""
        # Mark invalid where dbp >= sbp or outside physiological ranges
        invalid_bp = (
            (df['dbp'] >= df['sbp']) | 
            ~df['sbp'].between(*self.vital_ranges['sbp']) |
            ~df['dbp'].between(*self.vital_ranges['dbp'])
        )
        
        # Set invalid readings to NaN
        df.loc[invalid_bp, ['sbp', 'dbp']] = np.nan
        
        return df


class FeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.features = FEATURE_SETS

    def engineer_features(self):
        """Main method to engineer all features"""
        df = self.data.copy()
        
        # Add clinical metrics first (needed for some categories)
        df = self._add_clinical_metrics(df)
        
        # Add vital sign categories
        df = self._add_vital_categories(df)
        
        # Add demographic features
        df = self._add_demographic_features(df)
        
        # Add clinical scores
        df = self._add_clinical_scores(df)
        
        # Add temporal features
        df = self._add_temporal_features(df)
        
        return df

    def _add_clinical_metrics(self, df):
        """Add calculated clinical metrics"""
        # All metrics as float
        df['map'] = df['dbp'] + (df['sbp'] - df['dbp'])/3
        df['pulse_pressure'] = df['sbp'] - df['dbp']
        df['shock_index'] = df['heartrate'] / df['sbp']
        df['rate_pressure_product'] = df['heartrate'] * df['sbp']
        df['temp_delta'] = df['temperature'] - 98.6
        
        return df

    def _add_vital_categories(self, df):
        """Add categorical features for vital signs"""
        # Binary features as int (1/0), with NA handling
        df['is_tachycardic'] = df['heartrate'].gt(100).fillna(False).astype(int)
        df['is_bradycardic'] = df['heartrate'].lt(60).fillna(False).astype(int)
        df['is_hypoxic'] = df['o2sat'].lt(92).fillna(False).astype(int)
        df['is_hypertensive'] = df['sbp'].gt(140).fillna(False).astype(int)
        df['is_hypotensive'] = df['sbp'].lt(90).fillna(False).astype(int)
        df['is_febrile'] = df['temperature'].gt(99.5).fillna(False).astype(int)
        df['is_hypothermic'] = df['temperature'].lt(97.0).fillna(False).astype(int)
        
        # Categorical features with np.nan for missing
        df['hr_category'] = df['heartrate'].apply(self._categorize_hr)
        df['resp_category'] = df['resprate'].apply(self._categorize_resp)
        df['o2_category'] = df['o2sat'].apply(self._categorize_pulseox)
        df['sbp_category'] = df['sbp'].apply(self._categorize_bp)
        df['dbp_category'] = df['dbp'].apply(self._categorize_dbp)
        df['temp_category'] = df['temperature'].apply(self._categorize_temp)
        df['pain_category'] = df['pain'].apply(self._categorize_pain)
        
        return df

    def _add_demographic_features(self, df):
        """Add demographic-based features"""
        df['age_group'] = df['age_at_ed'].apply(self._categorize_age)
        df['acuity_group'] = df['acuity'].apply(self._categorize_acuity)
        return df

    def _add_clinical_scores(self, df):
        """Add clinical scoring features"""
        # Binary feature as int (1/0)
        df['has_sirs'] = df.apply(
            lambda x: self._calculate_sirs(
                x['temperature'], 
                x['heartrate'], 
                x['resprate']
            ), axis=1
        ).astype(int)
        return df

    def _add_temporal_features(self, df):
        """Add time-based features"""
        # Binary feature as int (1/0)
        df['is_daytime'] = (
            (df['hour'] >= 7) & (df['hour'] < 19)
        ).fillna(False).astype('Int64')
        
        # Calculate both weekly and daily minutes
        mask = df['dow'].notna() & df['hour'].notna() & df['minute'].notna()
        df['week_minutes'] = np.nan  # initialize with NaN
        df['day_minutes'] = np.nan   # initialize daily minutes
        
        if mask.any():
            # Map full day names to numbers (Monday = 0)
            day_map = {
                'Monday': 0,
                'Tuesday': 1,
                'Wednesday': 2,
                'Thursday': 3,
                'Friday': 4,
                'Saturday': 5,
                'Sunday': 6
            }
            
            # Calculate weekly minutes for valid rows
            days = df.loc[mask, 'dow'].map(day_map)
            week_minutes = (
                days * 24 * 60 +  # days to minutes
                df.loc[mask, 'hour'] * 60 +  # hours to minutes
                df.loc[mask, 'minute']  # minutes
            )
            
            # Calculate daily minutes (resets each day)
            day_minutes = (
                df.loc[mask, 'hour'] * 60 +  # hours to minutes
                df.loc[mask, 'minute']  # minutes
            )
            
            # Assign values
            df.loc[mask, 'week_minutes'] = week_minutes.round()
            df.loc[mask, 'day_minutes'] = day_minutes.round()
        
        # Convert to nullable integer type
        df['week_minutes'] = df['week_minutes'].astype('Int64')
        df['day_minutes'] = df['day_minutes'].astype('Int64')
        
        return df

    def _categorize_hr(self, hr):
        if pd.isna(hr): return np.nan
        if hr < 60: return "bradycardic"
        if hr > 100: return "tachycardic"
        return "normal"

    def _categorize_resp(self, resp):
        if pd.isna(resp): return np.nan
        if resp < 12: return "low"
        if resp > 20: return "high"
        return "normal"

    def _categorize_pulseox(self, po):
        if pd.isna(po): return np.nan
        return "low" if po < 92 else "normal"

    def _categorize_bp(self, bp):
        if pd.isna(bp): return np.nan
        if bp < 90: return "low"
        if bp > 140: return "high"
        return "normal"

    def _categorize_temp(self, temp):
        if pd.isna(temp): return np.nan
        if temp < 97.0: return "hypothermic"
        if temp > 99.5: return "febrile"
        return "normal"

    def _categorize_dbp(self, dbp):
        if pd.isna(dbp): return np.nan
        if dbp < 60: return "low"
        if dbp > 90: return "high"
        return "normal"

    def _categorize_pain(self, pain):
        if pd.isna(pain): return np.nan
        if pain == 0: return "no pain"
        if pain <= 3: return "mild"
        if pain <= 6: return "moderate"
        return "severe"

    def _categorize_acuity(self, acuity):
        try:
            return f"acuity_{int(acuity)}"
        except:
            return np.nan

    def _categorize_age(self, age):
        try:
            age = float(age)
        except:
            return np.nan
        if age < 18: return "child"
        if age < 36: return "young_adult"
        if age < 56: return "adult"
        if age < 76: return "middle_aged"
        return "senior"

    def _calculate_sirs(self, temp, hr, resp):
        if pd.isna(temp) or pd.isna(hr) or pd.isna(resp):
            return 0
        count = 0
        if (temp > 38) or (temp < 36): count += 1
        if hr > 90: count += 1
        if resp > 20: count += 1
        return 1 if count >= 2 else 0

class DataTransformer:
    def __init__(self, data):
        """
        Initialize DataTransformer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to transform
        """
        self.data = data
        self.features = FEATURE_SETS
        self.fitted_transformers = {}
        
        # Store transformation parameters
        self.boxcox_lambdas = {}
        self.feature_ranges = {}
        self.validation_results = {}
        self.scalers = {}
        
        # Define features that need transformation
        self.transform_features = {
            'age_at_ed': {'type': 'standard_scale', 'suffix': '_ss'},
            'temperature': {'type': 'beta', 'suffix': '_beta', 'range': [95.0, 108.0]},
            'temp_delta': {'type': 'standard_scale', 'suffix': '_ss'},
            'heartrate': {'type': 'standard_scale', 'suffix': '_ss'},
            'sbp': {'type': 'boxcox', 'suffix': '_bx'},
            'dbp': {'type': 'boxcox', 'suffix': '_bx'},
            'pulse_pressure': {'type': 'boxcox', 'suffix': '_bx'},
            'map': {'type': 'boxcox', 'suffix': '_bx'},
            'o2sat': {'type': 'invert_boxcox', 'suffix': '_ibx'},
            'shock_index': {'type': 'boxcox', 'suffix': '_bx'},
            'rate_pressure_product': {'type': 'boxcox', 'suffix': '_bx'},
            'resprate': {'type': 'boxcox', 'suffix': '_bx'},
            'pain': {'type': 'boxcox', 'suffix': '_bx'},
            'week_minutes': {'type': 'cyclical', 'suffix': '_cyc'},
            'day_minutes': {'type': 'cyclical', 'suffix': '_cyc'}
        }

    def fit_transform(self, data=None):
        """Fit transformer and transform data"""
        if data is not None:
            self.data = data.copy()
        
        df = self.data.copy()
        transformed_features = []

        # Process each feature according to its transformation type
        for feature, config in self.transform_features.items():
            if feature not in df.columns:
                continue

            transform_type = config['type']
            suffix = config['suffix']
            
            if transform_type == 'standard_scale':
                scaler = MinMaxScaler(feature_range=(-1, 1))
                valid_mask = ~df[feature].isna()
                df.loc[valid_mask, feature + suffix] = scaler.fit_transform(
                    df.loc[valid_mask, feature].values.reshape(-1, 1)
                )
                self.scalers[feature] = scaler
                transformed_features.append(feature + suffix)

            elif transform_type == 'boxcox':
                df[feature + suffix] = self._boxcox_transform(df[feature])
                transformed_features.append(feature + suffix)

            elif transform_type == 'beta':
                df[feature + suffix] = self._beta_transform(
                    df[feature], 
                    config['range'][0], 
                    config['range'][1]
                )
                transformed_features.append(feature + suffix)

            elif transform_type == 'invert_boxcox':
                df[feature + suffix] = self._transform_o2sat(df[feature])
                transformed_features.append(feature + suffix)

            elif transform_type == 'cyclical':
                sin, cos = self._cyclical_transform(df[feature])
                df[f'{feature}_sin'] = sin
                df[f'{feature}_cos'] = cos
                transformed_features.extend([f'{feature}_sin', f'{feature}_cos'])

        self.fitted = True
        self._validate_transformations(df, transformed_features)
        
        return df, self.features['gmm_features']

    def _beta_transform(self, series, min_val, max_val):
        """Normalize values to 0-1 range"""
        non_nan_mask = ~series.isna()
        transformed = pd.Series(index=series.index, dtype=float)
        
        if non_nan_mask.any():
            transformed[non_nan_mask] = (series[non_nan_mask] - min_val) / (max_val - min_val)
            self.feature_ranges[series.name] = (min_val, max_val)
            
        return transformed

    def _transform_o2sat(self, series):
        """Transform O2 saturation with inversion and Box-Cox"""
        non_nan_mask = ~series.isna()
        transformed = pd.Series(index=series.index, dtype=float)
        
        if non_nan_mask.any():
            # Add jitter to break discreteness
            jittered = series[non_nan_mask] + np.random.uniform(-0.5, 0.5, size=non_nan_mask.sum())
            
            # Invert values
            inverted = 100 - jittered
            
            # Beta transformation (normalize to 0-1)
            min_val = inverted.min()
            max_val = inverted.max()
            beta_scaled = (inverted - min_val) / (max_val - min_val) + 1e-10
            
            # Apply Box-Cox transformation
            boxcox_transformed, lambda_param = stats.boxcox(beta_scaled)
            self.boxcox_lambdas[series.name] = lambda_param
            
            # Scale to 0-1 range
            transformed[non_nan_mask] = (boxcox_transformed - boxcox_transformed.min()) / (
                boxcox_transformed.max() - boxcox_transformed.min()
            )
            
        return transformed

    def _boxcox_transform(self, series):
        """Apply Box-Cox transformation to strictly positive data"""
        non_nan_mask = ~series.isna()
        transformed = pd.Series(index=series.index, dtype=float)
        
        if non_nan_mask.any():
            x = series[non_nan_mask].copy()
            
            # Add jitter for discrete values
            if series.name == 'resprate':
                x = x + np.random.uniform(-1, 1, size=len(x))  # ±1 jitter for resprate
            elif series.name == 'pain':
                x = x + np.random.uniform(-0.5, 0.5, size=len(x))  # ±0.5 jitter for pain
            
            # Ensure strictly positive values
            if x.min() <= 0:
                x = x - x.min() + 1e-10
            
            try:
                transformed_data, lambda_param = stats.boxcox(x)
                self.boxcox_lambdas[series.name] = lambda_param
                
                # Scale to 0-1 range
                transformed[non_nan_mask] = (transformed_data - transformed_data.min()) / (
                    transformed_data.max() - transformed_data.min()
                )
            except Exception as e:
                print(f"Warning: Box-Cox failed for {series.name}, falling back to log transform")
                transformed[non_nan_mask] = np.log1p(x)
                self.boxcox_lambdas[series.name] = 0  # log transform
        
        return transformed

    def _cyclical_transform(self, series):
        """Transform cyclical features into sin and cos components"""
        minutes_in_week = 7 * 24 * 60
        non_nan_mask = ~series.isna()
        
        sin = pd.Series(index=series.index, dtype=float)
        cos = pd.Series(index=series.index, dtype=float)
        
        if non_nan_mask.any():
            sin[non_nan_mask] = np.sin(2 * np.pi * series[non_nan_mask] / minutes_in_week)
            cos[non_nan_mask] = np.cos(2 * np.pi * series[non_nan_mask] / minutes_in_week)
            
        return sin, cos

    def _validate_transformations(self, df, transformed_features):
        """Validate transformations using normality tests and plots"""
        from scipy import stats
        
        validation_results = {}
        for feature in transformed_features:
            if feature.endswith(('_sin', '_cos')):
                continue  # Skip cyclical components
                
            data = df[feature].dropna()
            if len(data) > 0:
                # Shapiro-Wilk test for normality
                statistic, p_value = stats.shapiro(data.sample(min(5000, len(data))))
                validation_results[feature] = {
                    'shapiro_statistic': statistic,
                    'shapiro_p_value': p_value,
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data)
                }
                
                if p_value < 0.05:
                    print(f"Warning: {feature} may not be normally distributed (p={p_value:.4f})")
                if abs(stats.skew(data)) > 1:
                    print(f"Warning: {feature} shows significant skewness ({stats.skew(data):.2f})")
        
        self.validation_results = validation_results
        return validation_results

    def get_params(self):
        """Return fitted parameters"""
        if not self.fitted:
            raise ValueError("Transformer not yet fitted")
        return {
            'boxcox_lambdas': self.boxcox_lambdas,
            'feature_ranges': self.feature_ranges,
            'validation_results': self.validation_results
        }

class PCATransformer:
    def __init__(self, data: pd.DataFrame, feature_weights=None):
        """
        Initialize PCATransformer
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to transform
        feature_weights : dict, optional
            Dictionary of feature weights to apply during PCA
        """
        self.data = data
        self.feature_weights = feature_weights if feature_weights else {}
        self.pca = None
        self.transformed_data = None

    def fit_transform(self, n_components=8, drop_missing=True):
        """
        Fit PCA and transform the data
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to retain
        drop_missing : bool, optional
            Whether to drop rows with missing values
        
        Returns:
        --------
        pd.DataFrame
            Transformed data after PCA
        """
        # Handle missing values
        if drop_missing:
            X = self.data.dropna()
        else:
            X = self.data.fillna(self.data.mean())  # Example imputation

        # Apply weights to the features
        weights = pd.Series(1.0, index=X.columns)
        for feature, weight in self.feature_weights.items():
            if feature in weights.index:
                weights[feature] = weight

        # Apply weights to the data
        X_weighted = X * weights

        # Initialize and fit PCA
        self.pca = PCA(n_components=n_components)
        self.transformed_data = self.pca.fit_transform(X_weighted)

        return pd.DataFrame(self.transformed_data, columns=[f'PC{i+1}' for i in range(self.pca.n_components_)])

    def optimize_feature_weights(self, target):
        """
        Optimize feature weights based on feature importance from a Random Forest model.
        
        Parameters:
        -----------
        target : pd.Series
            The target variable for supervised learning.
        
        Returns:
        --------
        dict
            Optimized feature weights
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.data, target, test_size=0.2, random_state=42)

        # Fit a Random Forest model
        model = RandomForestRegressor()  # or RandomForestClassifier for classification tasks
        model.fit(X_train, y_train)

        # Get feature importances
        importances = model.feature_importances_

        # Create a dictionary of feature weights
        optimized_weights = {feature: importance for feature, importance in zip(X_train.columns, importances)}

        # Normalize weights
        total_weight = sum(optimized_weights.values())
        for feature in optimized_weights:
            optimized_weights[feature] /= total_weight

        return optimized_weights

    def plot_cumulative_variance(self):
        """
        Plot cumulative explained variance to help select n_components
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Please call fit_transform first.")

        exp_var_ratio = self.pca.explained_variance_ratio_
        cum_exp_var_ratio = np.cumsum(exp_var_ratio)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cum_exp_var_ratio) + 1), cum_exp_var_ratio, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        plt.axhline(y=0.95, color='g', linestyle='--')  # Example threshold
        plt.show()

    def plot_explained_variance(self):
        """
        Plot the explained variance ratio of the PCA components
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Please call fit_transform first.")

        exp_var_ratio = self.pca.explained_variance_ratio_
        cum_exp_var_ratio = np.cumsum(exp_var_ratio)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(exp_var_ratio) + 1), exp_var_ratio, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cum_exp_var_ratio) + 1), cum_exp_var_ratio, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance')
        plt.tight_layout()
        plt.show()

    def get_pca_results(self):
        """
        Get PCA results including components and explained variance
        
        Returns:
        --------
        dict
            Dictionary containing PCA components and explained variance
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Please call fit_transform first.")

        return {
            'components': self.pca.components_,
            'explained_variance': self.pca.explained_variance_ratio_
        }