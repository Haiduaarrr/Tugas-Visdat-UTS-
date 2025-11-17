import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==============================
# Konfigurasi Halaman & Tema Warna
# ==============================
st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema warna
COLORS = {
    'primary': '#96cff1',
    'secondary': '#6dc0f1', 
    'accent': '#43b0f1',
    'dark_blue': '#057dcd',
    'navy': '#1e3d58',
    'background': '#f0f8ff',
    'orange': '#f4a261',
    'purple': '#9b5de5',
    'green': '#90EE90',    
    'red': '#FF6B6B'        
}


# CSS Custom dengan tema warna
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {COLORS['navy']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }}
    .section-header {{
        color: {COLORS['dark_blue']};
        border-bottom: 2px solid {COLORS['accent']};
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .info-box {{
        background-color: {COLORS['background']};
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['accent']};
        margin: 1rem 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['primary']};
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        margin-right: 2px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['dark_blue']} !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==============================
# Fungsi Helper
# ==============================
def load_and_preprocess_data():
    """Load dan preprocessing data"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")
        df_raw = pd.read_csv(path + '/hotel_bookings.csv')
        
        # Preprocessing awal
        df_raw = df_raw.drop(['agent', 'company'], axis=1)
        df_raw = df_raw.dropna()

        # Rename kolom
        df_raw = df_raw.rename(columns={
            'hotel': 'hotel_type',
            'is_canceled': 'canceled',
            'lead_time': 'lead_time',
            'arrival_date_year': 'arrival_year',
            'arrival_date_month': 'arrival_month',
            'arrival_date_week_number': 'arrival_week',
            'arrival_date_day_of_month': 'arrival_day',
            'stays_in_weekend_nights': 'weekend_nights',
            'stays_in_week_nights': 'week_nights',
            'adults': 'adults',
            'children': 'children',
            'babies': 'babies',
            'meal': 'meal_type',
            'country': 'country',
            'market_segment': 'market_segment',
            'distribution_channel': 'channel',
            'is_repeated_guest': 'repeated_guest',
            'previous_cancellations': 'prev_cancel',
            'previous_bookings_not_canceled': 'prev_book',
            'reserved_room_type': 'reserved_room',
            'assigned_room_type': 'assigned_room',
            'booking_changes': 'changes',
            'deposit_type': 'deposit',
            'days_in_waiting_list': 'waiting_days',
            'customer_type': 'customer_type',
            'adr': 'avg_daily_rate',
            'required_car_parking_spaces': 'parking_spaces',
            'total_of_special_requests': 'special_requests',
            'reservation_status': 'status',
            'reservation_status_date': 'status_date'
        })

        # Simpan versi mentah (belum capping)
        df_before_capping = df_raw.copy()

        # Tambahkan kolom total malam
        df_raw['total_nights'] = df_raw['weekend_nights'] + df_raw['week_nights']

        # Handle outlier numerik (IQR capping)
        df_capped = df_raw.copy()
        numeric_cols = df_capped.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - 1.5 * IQR
            upper_cap = Q3 + 1.5 * IQR
            df_capped[col] = np.where(df_capped[col] < lower_cap, lower_cap,
                               np.where(df_capped[col] > upper_cap, upper_cap, df_capped[col]))

        return df_before_capping, df_capped

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def create_metric_card(value, label, delta=None):
    """Membuat metric card dengan tema warna"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label=label, value=value, delta=delta)
    return col1

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.markdown(f"<h2 style='color: {COLORS['navy']};'>🏨 Hotel Analytics</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigasi", ["📊 Data Statistik", "🔍 Proses EDA", "📈 Visualisasi"])

# Load data sekali saja
if 'df_raw' not in st.session_state:
    st.session_state.df_raw, st.session_state.df = load_and_preprocess_data()

df_raw = st.session_state.df_raw
df = st.session_state.df

if df is None:
    st.error("Gagal memuat data. Pastikan kagglehub terinstall dan koneksi internet tersedia.")
    st.stop()

# ==============================
# Halaman 1: Data Statistik
# ==============================
if page == "📊 Data Statistik":
    st.markdown(f"<h1 class='main-header'>📊 Data Statistik Hotel Booking</h1>", unsafe_allow_html=True)
    
    # Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bookings", f"{len(df):,}")
    with col2:
        cancellation_rate = f"{(df['canceled'].mean() * 100):.1f}%"
        st.metric("Cancellation Rate", cancellation_rate)
    with col3:
        st.metric("Average Lead Time", f"{df['lead_time'].mean():.1f} days")
    with col4:
        st.metric("Unique Countries", df['country'].nunique())
    
    # Tabs untuk berbagai statistik
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Descriptive Stats", "🔍 Missing Values", "📊 Data Types"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.subheader("Hotel Distribution")
            hotel_counts = df['hotel_type'].value_counts()
            fig = px.pie(values=hotel_counts.values, names=hotel_counts.index,
                        color_discrete_sequence=[COLORS['primary'], COLORS['accent']])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Descriptive Statistics - Numerical Variables")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("Descriptive Statistics - Categorical Variables")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Tampilkan 5 pertama saja
            st.write(f"**{col}:**")
            st.write(df[col].value_counts().head())
    
    with tab3:
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ Tidak ada missing values dalam dataset")
        else:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualisasi missing values
            fig = px.bar(missing_df, x='Column', y='Missing Values', 
                        title='Missing Values Distribution',
                        color='Missing Values',
                        color_continuous_scale=[COLORS['primary'], COLORS['dark_blue']])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Data Types Overview")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index.astype(str),
                    title='Distribution of Data Types',
                    color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent']])
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan detail tipe data
        st.subheader("Detailed Data Types")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.notnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_info, use_container_width=True)

# ==============================
# Halaman 2: Proses EDA
# ==============================
elif page == "🔍 Proses EDA":
    st.markdown(f"<h1 class='main-header'>🔍 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Data Cleaning", 
        "📈 Outlier Analysis",  
        "📋 Feature Engineering"
    ])
    
    # =============================
    # TAB 1: Data Cleaning
    # =============================
    with tab1:
        st.header("📊 Data Cleaning Process")
        
        st.subheader("1. Handling Missing Values")
        st.markdown("""
        **Kolom yang dihapus:**
        - `agent`: 13.69% missing values  
        - `company`: 94.31% missing values  

        **Tindakan:** Drop kolom karena persentase missing values terlalu tinggi.
        """)
        
        st.subheader("2. Data Type Conversion")
        st.markdown("""
        **Renaming columns untuk konsistensi:**
        - `hotel` → `hotel_type`  
        - `is_canceled` → `canceled`  
        - `adr` → `avg_daily_rate`  
        - `stays_in_weekend_nights` → `weekend_nights`  
        - `stays_in_week_nights` → `week_nights`
        """)
        
        st.subheader("3. Feature Engineering")
        st.markdown("""
        **Kolom baru yang dibuat:**
        - `total_nights` = `weekend_nights` + `week_nights`
        """)

    # =============================
    # TAB 2: Outlier Analysis
    # =============================
    with tab2:
        st.header("📈 Outlier Analysis")

        # Daftar kolom continuous
        continuous_vars = [
            'changes', 'waiting_days', 'adults', 'week_nights', 'prev_cancel',
            'prev_book', 'arrival_week', 'arrival_day', 'avg_daily_rate',
            'lead_time', 'weekend_nights'
        ]

        selected_col = st.selectbox("Pilih kolom untuk analisis outlier:", continuous_vars)

        if selected_col:
            # Hitung batas IQR
            Q1 = df_raw[selected_col].quantile(0.25)
            Q3 = df_raw[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # ====== Statistik Sebelum Handling ======
            outliers = df_raw[(df_raw[selected_col] < lower_bound) | (df_raw[selected_col] > upper_bound)]
            st.markdown(f"""
            **Statistik Outlier ({selected_col})**
            - Jumlah Outlier: `{len(outliers):,}`
            - Persentase Outlier: `{(len(outliers) / len(df_raw) * 100):.2f}%`
            - Lower Bound: `{lower_bound:.2f}`
            - Upper Bound: `{upper_bound:.2f}`
            """)

            # ====== Layout Before dan After sejajar ======
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sebelum Handling Outlier")
                fig_before = px.box(
                    df_raw,
                    y=selected_col,
                    title=f"Distribusi {selected_col} (Before Handling)",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_before, use_container_width=True)

            with col2:
                st.subheader("Setelah Handling Outlier (IQR Capping)")
                df_capped = df_raw.copy()
                df_capped[selected_col] = np.where(
                    df_capped[selected_col] < lower_bound, lower_bound, df_capped[selected_col]
                )
                df_capped[selected_col] = np.where(
                    df_capped[selected_col] > upper_bound, upper_bound, df_capped[selected_col]
                )

                fig_after = px.box(
                    df_capped,
                    y=selected_col,
                    title=f"Distribusi {selected_col} (After Handling)",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_after, use_container_width=True)

            # ====== Penjelasan Metode ======
            st.markdown("""
            ### 🧮 Metode IQR Capping
            Outlier diatasi dengan mengganti nilai ekstrem agar tidak memengaruhi model secara signifikan.

            **Rumus:**
            - Q1 = Kuartil 1 (25%)
            - Q3 = Kuartil 3 (75%)
            - IQR = Q3 - Q1  
            - Lower Bound = Q1 - 1.5 × IQR  
            - Upper Bound = Q3 + 1.5 × IQR  

            Nilai di bawah *Lower Bound* diganti menjadi *Lower Bound*,  
            nilai di atas *Upper Bound* diganti menjadi *Upper Bound*.
            """)

    # # =============================
    # # TAB 3: Correlation Analysis
    # # =============================
    # with tab3:
    #     st.header("🔍 Correlation Analysis")
        
    #     numeric_df = df.select_dtypes(include=[np.number])
    #     correlation_matrix = numeric_df.corr()
        
    #     fig = px.imshow(
    #         correlation_matrix,
    #         title="Correlation Matrix Heatmap",
    #         color_continuous_scale=[COLORS['primary'], 'white', COLORS['dark_blue']],
    #         aspect="auto"
    #     )
    #     st.plotly_chart(fig, use_container_width=True)
        
    #     # Top Positive and Negative Correlations
    #     corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    #     top_positive = corr_pairs[corr_pairs < 1].head(10)
    #     top_negative = corr_pairs.tail(10)

    #     st.subheader("Top Positive Correlations")
    #     st.dataframe(top_positive.reset_index().rename(columns={
    #         'level_0': 'Var1', 'level_1': 'Var2', 0: 'Correlation'
    #     }), use_container_width=True)
        
    #     st.subheader("Top Negative Correlations")
    #     st.dataframe(top_negative.reset_index().rename(columns={
    #         'level_0': 'Var1', 'level_1': 'Var2', 0: 'Correlation'
    #     }), use_container_width=True)
    
    # =============================
    # TAB 3: Feature Engineering
    # =============================
    with tab3:
        st.header("📋 Feature Engineering & Variable Identification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Continuous Variables")
            continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
            for var in continuous_vars[:10]:
                st.write(f"• {var}")
            
            st.subheader("Key Derived Features")
            st.markdown("""
            - **total_nights**: Total lama menginap  
            - **cancellation_rate**: Tingkat pembatalan per segment  
            - **booking_lead_ratio**: Rasio lead time terhadap ADR
            """)
        
        with col2:
            st.subheader("Categorical Variables")
            categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
            for var in categorical_vars[:10]:
                unique_count = df[var].nunique()
                st.write(f"• {var} ({unique_count} unique values)")
            
            st.subheader("Target Variable")
            st.markdown("""
            **canceled**:  
            - 0: Booking tidak dibatalkan  
            - 1: Booking dibatalkan
            """)

# ==============================
# HALAMAN 3: VISUALISASI + ANALISIS FINAL 
# ==============================
if page == "📈 Visualisasi":
    st.markdown(f"<h1 class='main-header'>📈 Visualisasi Analisis Hotel Booking</h1>", unsafe_allow_html=True)
    st.info("Berikut 10 analisis visualisasi lengkap dengan tujuan, hasil analisis, dan alasan pemilihan chart.")

    # Pastikan tipe data aman
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # =========================================================
    # 1. Sunburst Chart – Pola Pembatalan Berdasarkan Jenis Hotel dan Bulan
    # =========================================================
    st.subheader("🌞 1. Sunburst Chart – Pola Pembatalan Berdasarkan Jenis Hotel dan Bulan")

    cancel_by_hotel_month = df.groupby(['hotel_type', 'arrival_month', 'canceled']).size().reset_index(name='count')
    cancel_hierarchy = cancel_by_hotel_month[cancel_by_hotel_month['canceled'] == 1]
    top_hotel = cancel_hierarchy.groupby('hotel_type')['count'].sum().idxmax()

    fig1 = px.sunburst(cancel_hierarchy, path=['hotel_type', 'arrival_month'], values='count',
                       color='hotel_type', color_discrete_sequence=[COLORS['primary'], COLORS['orange']],
                       title='Pola Pembatalan Berdasarkan Jenis Hotel dan Bulan')
    fig1.add_annotation(text=f"{top_hotel} memiliki pembatalan terbanyak",
                        x=0.95, y=0.95, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12, color="yellow"))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Melihat pola pembatalan yang bergantung pada dua faktor hierarkis: jenis hotel dan bulan kedatangan.  

    **Analisis:**  
    - City Hotel memiliki proporsi pembatalan yang jauh lebih besar dibanding Resort Hotel.  
    - Pembatalan paling banyak terjadi pada bulan-bulan puncak (musim liburan), menunjukkan adanya pengaruh musiman terhadap perilaku tamu.  
    """)
    with st.expander("📊 Kenapa Sunburst Chart:"):
        st.markdown("""
        - Sunburst efektif untuk menampilkan relasi hierarkis antar kategori (Hotel → Bulan).  
        - Warna dan ukuran segmen langsung menunjukkan proporsi pembatalan secara intuitif.
        """)

    # =========================================================
    # 2. Scatter Chart – Hubungan Lead Time dan Average Daily Rate terhadap Pembatalan
    # =========================================================

    st.subheader("📈 2. Scatter Chart – Lead Time vs Average Daily Rate (Dipisah per Pembatalan)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❌ Tidak Dibatalkan (0)")
        fig2a = px.scatter(
            df[df['canceled'] == 0],
            x='lead_time', y='avg_daily_rate',
            title="Scatter – Pesanan Tidak Dibatalkan",
            color_discrete_sequence=["#3498db"],  # biru
            hover_data=['hotel_type']
        )
        st.plotly_chart(fig2a, use_container_width=True)

    with col2:
        st.markdown("### ✅ Dibatalkan (1)")
        fig2b = px.scatter(
            df[df['canceled'] == 1],
            x='lead_time', y='avg_daily_rate',
            title="Scatter – Pesanan Dibatalkan",
            color_discrete_sequence=["#ff8888"],  # merah
            hover_data=['hotel_type']
        )
        st.plotly_chart(fig2b, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Menganalisis hubungan antara lamanya waktu pemesanan sebelum check-in (lead time) dan harga rata-rata kamar terhadap kemungkinan pembatalan.  

    **Analisis:**  
    - Pesanan dengan lead time tinggi cenderung memiliki tingkat pembatalan yang lebih besar.  
    - Harga kamar (ADR) tidak menunjukkan pengaruh signifikan terhadap pembatalan karena sebaran titik relatif merata di semua rentang harga.  
    """)
    with st.expander("📊 Kenapa Scatter Chart:"):
        st.markdown("""
        - Cocok untuk mengeksplorasi hubungan antara dua variabel numerik.  
        - Warna digunakan sebagai dimensi ketiga untuk menunjukkan status pembatalan (dibatalkan atau tidak).
        """)

    # =========================================================
    # 3. Violin Plot – Distribusi Lama Menginap antara City dan Resort Hotel
    # =========================================================
    st.subheader("🎻 3. Violin Plot – Distribusi Lama Menginap antara City dan Resort Hotel")

    fig3 = px.violin(df, x='hotel_type', y='total_nights', color='hotel_type',
                     box=True, points='all',
                     color_discrete_sequence=[COLORS['orange'], COLORS['purple']],
                     title='Distribusi Lama Menginap antara City dan Resort Hotel')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Membandingkan distribusi lama menginap tamu pada dua jenis hotel yang berbeda.  

    **Analisis:**  
    - Resort Hotel memiliki distribusi lama menginap yang lebih lebar, menandakan banyak tamu menginap lebih dari 5 malam.  
    - City Hotel cenderung memiliki lama menginap yang singkat (1–3 malam), mencerminkan tamu bisnis atau perjalanan singkat.  
    """)
    with st.expander("📊 Kenapa Violin Plot:"):
        st.markdown("""
        - Menampilkan distribusi dan kepadatan data secara bersamaan, lebih informatif dibanding box plot biasa.  
        - Dapat memperlihatkan median, outlier, dan sebaran secara visual dalam satu grafik.
        """)

    # =========================================================
    # 4. Choropleth Map – Tingkat Pembatalan Berdasarkan Negara Asal
    # =========================================================
    st.subheader("🌍 4. Choropleth Map – Tingkat Pembatalan Berdasarkan Negara Asal")
    cancel_by_country = df.groupby('country')['canceled'].mean().reset_index()
    gap = px.data.gapminder()[['iso_alpha', 'country', 'continent']].drop_duplicates()
    gap = gap.rename(columns={'country': 'country_full_name'})
    merged_df = cancel_by_country.merge(
        gap,
        left_on='country',
        right_on='iso_alpha',
        how='left'
    )
    # === Filter Benua ===
    continents = merged_df['continent'].dropna().unique()
    selected_continent = st.selectbox("🌎 Pilih Benua:", options=continents)
    filtered_map = merged_df[merged_df['continent'] == selected_continent]
    top_country = filtered_map.sort_values('canceled', ascending=False).head(1)
    fig4 = px.choropleth(
        filtered_map,
        locations='country',
        locationmode='ISO-3',
        color='canceled',
        hover_name='country_full_name',
        custom_data=['country_full_name'],
        projection='natural earth',
        color_continuous_scale='Reds',
        title=f"Tingkat Pembatalan Berdasarkan Negara Asal – {selected_continent}",
        labels={'canceled': 'Tingkat Pembatalan'}
    )

    fig4.update_layout(coloraxis_colorbar_tickformat=':.1%')
    fig4.update_traces(
        hovertemplate='<b>%{customdata[0]} (%{location})</b><br>Tingkat Pembatalan: %{z:.2%}<extra></extra>'
    )
    tc = top_country.iloc[0]
    fig4.add_annotation(
        text=f"Tertinggi: {tc['country_full_name']} ({tc['country']})<br>{tc['canceled']:.1%}",
        x=0.95, y=0.95, xref="paper", yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.7)",
        font=dict(color="black", size=14) )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(f"""
    **Tujuan:** Menunjukkan variasi tingkat pembatalan berdasarkan negara asal tamu untuk memahami pola geografis yang memengaruhi perilaku pembatalan. 

    **Analisis:** 
    - Tingkat pembatalan berbeda-beda antar negara, menunjukkan adanya faktor geografis, sosial, dan ekonomi yang memengaruhi keputusan pembatalan.
    - Beberapa negara menampilkan tingkat pembatalan tinggi, yang bisa disebabkan oleh akses transportasi, biaya perjalanan, ketidakpastian jadwal, atau regulasi perjalanan.
    - Negara dengan akses perjalanan yang lebih dekat dan stabil cenderung memiliki tingkat pembatalan lebih rendah.
    """)

    with st.expander("📊 Kenapa Choropleth Map:"):
        st.markdown("""
        - Memudahkan pengamatan pola geografis dari data pembatalan.  
        - Warna menggambarkan intensitas pembatalan antarnegara secara langsung dan intuitif.
        """)

    # =========================================================
    # 5. Treemap Interaktif – Struktur Pembatalan Berdasarkan Channel dan Tipe Pelanggan
    # =========================================================
    st.subheader("🧩 5. Treemap – Struktur Pembatalan Berdasarkan Channel & Customer Type")
    clean_df = df[
        df['channel'].notna() &
        (df['channel'] != 'Undefined') &
        df['customer_type'].notna() &
        (df['customer_type'] != 'Undefined')
    ]
    cancel_treemap = (
        clean_df.groupby(['channel', 'customer_type'])['canceled']
        .mean()
        .reset_index()
    )
    fig5 = px.treemap(
        cancel_treemap,
        path=['channel', 'customer_type'],
        values='canceled',
        color='canceled',
        color_continuous_scale='RdPu',
        title="📊 Struktur Pembatalan Berdasarkan Channel & Customer Type",
        hover_data={'canceled': ':.2f'}
    )
    fig5.update_traces(
        textinfo='label+value+percent parent',
        texttemplate="<b>%{label}</b><br>Rasio: %{value:.2f}<br>(%{percentParent:.1%})",
        textfont_size=13,
        marker=dict(line=dict(width=2, color='white'))
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Mengetahui kontribusi dan kecenderungan pembatalan pemesanan berdasarkan *saluran distribusi (channel)* serta *tipe pelanggan (customer type)*.  
    Visualisasi ini membantu memahami bagaimana pola pembatalan berbeda antara pelanggan individual, grup, dan kontrak melalui berbagai jalur pemesanan.

    **Analisis:** 
    - Channel **TA/TO (Travel Agent/Tour Operator)** mendominasi proporsi pembatalan secara keseluruhan.  
    Dalam channel ini, pelanggan **Transient** menunjukkan tingkat pembatalan **tertinggi**, mencerminkan fleksibilitas tinggi tamu individu.  
    - Channel **Corporate** menunjukkan tingkat pembatalan **menengah**, terutama pada segmen **Contract** dan **Transient-Party**, yang sering kali terkait dengan perjalanan bisnis.  
    - Channel **Direct** memiliki tingkat pembatalan **paling rendah**, khususnya pada pelanggan **Contract** dan **Group**, yang menunjukkan loyalitas serta komitmen lebih kuat terhadap pemesanan langsung.  
    - Channel **GDS** memiliki volume kecil namun stabil, dengan pembatalan rendah pada pelanggan **Transient**.  
    """)


    with st.expander("📊 Kenapa Treemap:"):
        st.markdown("""
        - Menampilkan **hubungan hierarkis** antara channel dan tipe pelanggan secara intuitif.  
        - Ukuran area menunjukkan **kontribusi volume pembatalan**, sedangkan warna menunjukkan **tingkat pembatalan relatif**.  
        - Efektif untuk memahami **struktur risiko pembatalan** dalam sistem distribusi hotel.  
        """)


    # =========================================================
    # 6. Stacked Vertical Bar Chart – Pembatalan Berdasarkan Jenis Deposit 
    # =========================================================
    st.subheader("🟠 6. Stacked Vertical Bar Chart – Pembatalan Berdasarkan Jenis Deposit")
    df_bar = df.copy()
    df_bar = df_bar[df_bar['deposit'].notna()]
    df_bar = df_bar[df_bar['deposit'] != "undefined"]
    df_bar['canceled_label'] = df_bar['canceled'].map({0: 'Not Canceled', 1: 'Canceled'})
    deposit_cancel = (
        df_bar.groupby(['deposit', 'canceled_label'])
        .size()
        .reset_index(name='count'))
    fig6 = px.bar(
        deposit_cancel,
        x='deposit',
        y='count',
        color='canceled_label',
        barmode='stack',
        title='Jumlah Pemesanan per Jenis Deposit & Status Pembatalan',
        text='count',
        labels={
            'deposit': 'Jenis Deposit',
            'count': 'Jumlah Pemesanan',
            'canceled_label': 'Status Pembatalan'
        },
        color_discrete_map={
            'Not Canceled': '#57D68D',
            'Canceled': '#FF6B6B'
        })
    fig6.update_traces(
        textposition='outside',
        textfont=dict(color='white', size=12))
    fig6.update_layout(
        width=900,
        height=600,
        font=dict(color='white', size=14),
        title_font=dict(color='white', size=18),
        legend=dict(font=dict(color='white')),
        xaxis={'categoryorder': 'total ascending'},
        yaxis=dict(
            rangemode='tozero',
            title='Jumlah Data',
            tickformat=',d'))
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Mengetahui distribusi pembatalan pemesanan berdasarkan jenis deposit yang diterapkan oleh hotel.  
    Visualisasi ini digunakan untuk mengidentifikasi hubungan antara kebijakan deposit dan kecenderungan pembatalan oleh pelanggan.  

    **Analisis:**  
    - Kategori No Deposit menunjukkan tingkat pembatalan tertinggi dibandingkan dengan kategori deposit lainnya.
    - Ketiadaan kewajiban pembayaran di awal memberi keleluasaan bagi pelanggan untuk membatalkan pemesanan tanpa risiko finansial.
    - Sebaliknya, kategori Non Refund memperlihatkan tingkat pembatalan paling rendah karena adanya konsekuensi finansial jika terjadi pembatalan.
    - Temuan ini menegaskan bahwa kebijakan deposit memiliki pengaruh signifikan terhadap perilaku pembatalan pelanggan.
    - Kebijakan Refundable (deposit yang dapat dikembalikan) memiliki volume pemesanan total yang sangat kecil, tetapi merupakan kategori yang paling sehat dengan tingkat pembatalan paling rendah. Ini menunjukkan pelanggan yang memilih opsi ini adalah pelanggan serius yang menghargai fleksibilitas.

    """)

    with st.expander("📊 Kenapa pakai bar chart:"):
        st.markdown("""
        - Stacked bar chart memudahkan perbandingan proporsi Canceled vs Not Canceled dalam satu kategori secara langsung.
        - Bentuk vertikal memudahkan pembacaan karena jumlah pemesanan secara alami naik ke atas.
        - Lebih informatif untuk menampilkan total nilai sekaligus komposisi status pembatalan pada tiap jenis deposit.
        """)

    # =========================================================
    # 7. Line Chart – Tren Jumlah Pemesanan per Bulan per Jenis Hotel
    # =========================================================
    st.subheader("📅 7. Line Chart – Tren Jumlah Pemesanan per Bulan per Jenis Hotel")

    month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    monthly = df.groupby(['arrival_month', 'hotel_type']).size().reset_index(name='count')
    monthly['arrival_month'] = pd.Categorical(monthly['arrival_month'], categories=month_order, ordered=True)
    monthly = monthly.sort_values('arrival_month')  # 🔹 Tambahkan baris ini
    top_month = monthly.groupby('arrival_month')['count'].sum().idxmax()

    fig7 = px.line(
        monthly, x='arrival_month', y='count', color='hotel_type', markers=True,
        color_discrete_sequence=[COLORS['primary'], COLORS['orange']],
        title='Tren Jumlah Pemesanan per Bulan per Jenis Hotel'
    )
    fig7.add_annotation(
        text=f"Puncak di bulan {top_month}",
        x=0.95, y=0.95, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color="yellow")
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Menelusuri pola musiman dan tren waktu pada jumlah pemesanan hotel berdasarkan jenis hotel.  

    **Analisis:**  
    - City Hotel menunjukkan fluktuasi besar dengan puncak pemesanan di bulan pertengahan tahun (musim liburan).  
    - Resort Hotel memiliki tren yang lebih stabil, tetapi volume pemesanan lebih kecil secara keseluruhan.  
    """)
    with st.expander("📊 Kenapa Line Chart:"):
        st.markdown("""
        - Efektif untuk menampilkan perubahan nilai dari waktu ke waktu (time series).  
        - Garis memudahkan perbandingan tren antara dua kategori hotel.
        """)

    # =========================================================
    # 8. Pie Chart – Proporsi Jenis Pelanggan terhadap Total Pemesanan
    # =========================================================
    st.subheader("🥧 8. Pie Chart – Proporsi Jenis Pelanggan terhadap Total Pemesanan")

    customer_counts = df['customer_type'].value_counts().reset_index()
    customer_counts.columns = ['customer_type', 'count']

    fig8 = px.pie(
        customer_counts, names='customer_type', values='count',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title='Proporsi Jenis Pelanggan terhadap Total Pemesanan'
    )
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("""
    **Tujuan:**  
    Mengetahui komposisi pelanggan berdasarkan tipe, seperti *Transient*, *Group*, atau *Contract*.  

    **Analisis:**  
    - Mayoritas tamu merupakan pelanggan *Transient* (individu tanpa kontrak).  
    - Jenis pelanggan *Group* dan *Contract* memiliki proporsi kecil, menunjukkan sebagian besar pelanggan datang untuk perjalanan pribadi.  
    """)
    with st.expander("📊 Kenapa Pie Chart:"):
        st.markdown("""
        - Pie chart sangat mudah dipahami oleh orang awam: potongan besar berarti proporsi besar.  
        - Efektif menampilkan perbandingan sederhana antar kategori.
        """)


    # =========================================================
    # 9. Scatter Chart – Korelasi Harga Kamar dan Pembatalan
    # =========================================================
    st.subheader("💰 9. Scatter Chart – Korelasi Harga Kamar dan Pembatalan")  # Menambahkan judul chart

    # Buat scatter plot
    fig9 = px.scatter(
        df, x='avg_daily_rate', y='canceled', color='hotel_type',
        color_discrete_sequence=[COLORS['orange'], COLORS['accent']],
        title='Korelasi antara Harga Kamar dan Pembatalan'
    ) 
    # Tambahkan anotasi
    fig9.add_annotation(
        text="Hubungan lemah antara harga dan pembatalan",
        x=0.95, y=0.95, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color="white")
    ) 
    st.plotly_chart(fig9, use_container_width=True)    # Tampilkan chart

    corr_price = df['avg_daily_rate'].corr(df['canceled'])
    st.markdown(f"""
    **Tujuan:**  
    Menganalisis apakah harga kamar (ADR) memiliki hubungan dengan kemungkinan pembatalan.  

    **Analisis:**  
    - Korelasi antara harga kamar dan pembatalan sangat lemah (**{corr_price:.3f}**).  
    - Pembatalan tidak banyak dipengaruhi oleh harga, tetapi lebih pada faktor waktu pemesanan dan preferensi tamu.  
    """)
    # Tambahkan expander
    with st.expander("📊 Kenapa Scatter Chart:"):
        st.markdown("""
        - Menunjukkan hubungan linear atau pola penyebaran antara dua variabel numerik.  
        - Mudah mendeteksi pola atau outlier dari hubungan harga dan pembatalan.
        """)

    # =========================================================
    # 10. Box Plot – Perbandingan Harga Berdasarkan Jenis Hotel dan Jenis Makanan
    # =========================================================
    st.subheader("🍽️ 10. Box Plot – Perbandingan Harga Berdasarkan Jenis Hotel dan Jenis Makanan")  # Untuk judul chart

    # Buat box plot
    fig10 = px.box(
        df, x='meal_type', y='avg_daily_rate', color='hotel_type',
        color_discrete_sequence=['#FFA600', '#58508D'],
        title='Sebaran Harga per Malam Berdasarkan Jenis Makanan dan Jenis Hotel'
    )
    top_meal = df.groupby('meal_type')['avg_daily_rate'].mean().idxmax() # Hitung meal type dengan harga tertinggi
    # Tambahkan anotasi
    fig10.add_annotation(
        text=f"{top_meal} = harga tertinggi",
        x=0.95, y=0.95, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color="white")
    )
    st.plotly_chart(fig10, use_container_width=True)   # Tampilkan chart

    meal_mean = df.groupby('meal_type')['avg_daily_rate'].mean().sort_values(ascending=False)
    st.markdown(f"""
    **Tujuan:**  
    Membandingkan variasi harga kamar berdasarkan jenis layanan makanan dan tipe hotel.  

    **Analisis:**  
    - Jenis makanan “Half Board” memiliki rata-rata harga kamar tertinggi.  
    - Perbedaan antar meal type menunjukkan bahwa layanan makanan turut memengaruhi strategi harga kamar.  
    """) # Tambahkan analisis
    with st.expander("📊 Kenapa Box Plot:"):
        st.markdown("""
        - Memudahkan melihat sebaran, median, dan outlier antar kategori.  
        - Efektif untuk mendeteksi perbedaan signifikan antar kelompok secara visual.
        """)

    # ====== Footer ======
    st.markdown("---")
    st.markdown(f"<p style='text-align:center;color:{COLORS['navy']};'>Dibuat dengan ❤️ oleh Tim Visualisasi Data</p>", unsafe_allow_html=True)
