file_path = "train_set.csv"
df = pd.read_csv(file_path, dtype=str, low_memory=False)

df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

df.dropna(subset=['start_time', 'end_time'], inplace=True)

df.drop_duplicates(inplace=True)

df['hour'] = df['start_time'].dt.hour
saturation_pass_type = df.groupby(['passholder_type', 'hour']).size().reset_index(name='num_viajes')
saturation_pass_type['num_viajes'] = pd.to_numeric(saturation_pass_type['num_viajes'], errors='coerce').fillna(0)

station_saturation = df.groupby('start_station').size().reset_index(name='num_viajes')
station_saturation = station_saturation.sort_values(by='num_viajes', ascending=False)

percentiles = station_saturation['num_viajes'].quantile([0.5, 0.85]).values

def categorize_abc(value):
    if value > percentiles[1]:
        return 'A'
    elif value > percentiles[0]:
        return 'B'
    else:
        return 'C'

station_saturation['ABC_Category'] = station_saturation['num_viajes'].apply(categorize_abc)
df = df.merge(station_saturation[['start_station', 'ABC_Category']], on='start_station', how='left')
df.rename(columns={'ABC_Category': 'start_station_category'}, inplace=True)

end_station_saturation = df.groupby('end_station').size().reset_index(name='num_viajes')
end_station_saturation = end_station_saturation.sort_values(by='num_viajes', ascending=False)
end_station_saturation['ABC_Category'] = end_station_saturation['num_viajes'].apply(categorize_abc)
df = df.merge(end_station_saturation[['end_station', 'ABC_Category']], on='end_station', how='left')
df.rename(columns={'ABC_Category': 'end_station_category'}, inplace=True)

df_saturation_abc = df.groupby(['hour', 'passholder_type', 'start_station_category']).size().reset_index(name='num_viajes')

g = sns.FacetGrid(
    data=df_saturation_abc,
    col='passholder_type',
    hue='start_station_category',
    col_wrap=2,
    height=4,
    sharey=False
)
g.map_dataframe(sns.lineplot, x='hour', y='num_viajes', marker='o')
g.add_legend()
g.set_titles(col_template='Tipo de pase: {col_name}')
for ax in g.axes.flatten():
    ax.set_xticks(range(0, 24))
    ax.grid(True)
g.fig.suptitle('Saturación del servicio por categoría ABC, tipo de pase y hora', y=1.05)
plt.show()

df['year_month'] = df['start_time'].dt.to_period('M')
total_usage = df.groupby('year_month').size().reset_index(name='total_trips')

total_usage['date'] = total_usage['year_month'].apply(lambda x: x.to_timestamp())

total_usage = total_usage.sort_values(by='date')

total_usage['month_number'] = range(1, len(total_usage) + 1)

X = total_usage[['month_number']]
y = total_usage['total_trips'].astype(float)  # Asegurarnos de que sea numérico

model = LinearRegression()
model.fit(X, y)

num_future_months = 6
last_month_number = total_usage['month_number'].max()
future_months = np.array(range(last_month_number + 1, last_month_number + num_future_months + 1)).reshape(-1, 1)

future_dates = pd.date_range(
    start=total_usage['date'].max() + pd.DateOffset(months=1),
    periods=num_future_months,
    freq='M'
)

predictions = model.predict(future_months)

prediction_df = pd.DataFrame({
    'date': future_dates,
    'total_trips': predictions
})

plot_df = pd.concat([
    total_usage[['date', 'total_trips']],
    prediction_df[['date', 'total_trips']]
]).reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.lineplot(data=plot_df, x='date', y='total_trips', marker='o', color='blue', label='Datos Reales / Proyección')
plt.axvline(x=total_usage['date'].max(), color='gray', linestyle='--', label='Inicio de la Proyección')
plt.xlabel('Fecha (Año-Mes)')
plt.ylabel('Total de Viajes')
plt.title('Proyección de la Tendencia Global de Uso de Bicicletas Compartidas (Regresión Lineal)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()

plan_growth = df.groupby(['year_month', 'passholder_type']).size().reset_index(name='num_viajes')
plan_growth['num_viajes'] = pd.to_numeric(plan_growth['num_viajes'], errors='coerce').fillna(0)
plan_growth['year_month'] = plan_growth['year_month'].astype(str)

pass_types = plan_growth['passholder_type'].unique()

plt.figure(figsize=(12, 6))


for idx, pt in enumerate(pass_types):
    subset = plan_growth[plan_growth['passholder_type'] == pt].copy()
    subset['date'] = pd.to_datetime(subset['year_month'], format='%Y-%m')
    subset = subset.sort_values(by='date')
    subset['month_number'] = range(1, len(subset) + 1)
    sns.lineplot(data=subset, x='year_month', y='num_viajes',
                 marker='o', linestyle='-', color=palette[idx],
                 label=pt)
    if len(subset) > 1:
        X = subset[['month_number']]
        y = subset['num_viajes'].astype(float)
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        num_future_months = 12
        last_month_number = subset['month_number'].max()
        future_months = np.array(range(last_month_number + 1, last_month_number + num_future_months + 1)).reshape(-1, 1)
        predictions = lr_model.predict(future_months)
        last_date = subset['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=num_future_months, freq='M')
        future_year_month = future_dates.strftime('%Y-%m')
        proj_df = pd.DataFrame({
            'year_month': future_year_month,
            'num_viajes': predictions.flatten(),
            'passholder_type': pt
        })
        sns.lineplot(data=proj_df, x='year_month', y='num_viajes',
                     marker='o', linestyle='--', color=palette[idx],
                     label=f'Proyección {pt}')
plt.xlabel('Fecha (Año-Mes)')
plt.ylabel('Número de Viajes')
plt.title('Crecimiento de Planes con Proyección por Tipo de Pase')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Pase', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

growth_rates = {}
pass_types = plan_growth['passholder_type'].unique()

for pt in pass_types:
    subset = plan_growth[plan_growth['passholder_type'] == pt].copy()
    subset['date'] = pd.to_datetime(subset['year_month'], format='%Y-%m')
    subset = subset.sort_values(by='date')
    subset['month_number'] = range(1, len(subset) + 1)
    if len(subset) > 1:
        X = subset[['month_number']]
        y = subset['num_viajes'].astype(float)
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        growth_rates[pt] = slope
print("Pendientes (crecimiento mensual) por tipo de pase:")
for pt, slope in growth_rates.items():
    print(f"{pt}: {slope:.2f}")

max_growth = max(growth_rates, key=lambda k: growth_rates[k])
print(f"\nEl pase que más crece es: {max_growth} con una pendiente de {growth_rates[max_growth]:.2f} viajes por mes.")

df['hour'] = df['start_time'].dt.hour
df['weekday'] = df['start_time'].dt.weekday
df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

bins = [0, 5, 10, 20, 40, 60, 120, 240]  # Ajusta estos valores según la distribución de tu dataset
labels = ["0-5 min", "5-10 min", "10-20 min", "20-40 min", "40-60 min", "60-120 min", "120+ min"]

df["duration_category"] = pd.cut(df["duration_minutes"], bins=bins, labels=labels, include_lowest=True)

df_grouped = df.groupby(["duration_category", "passholder_type"]).size().reset_index(name="count")

plt.figure(figsize=(12, 6))
sns.barplot(data=df_grouped, x="duration_category", y="count", hue="passholder_type")

plt.xlabel("Intervalo de duración (minutos)")
plt.ylabel("Cantidad de viajes")
plt.title("Distribución de tipos de pase por duración del viaje")
plt.xticks(rotation=45)
plt.legend(title="Tipo de Pase")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

df['hour'] = df['start_time'].dt.hour
df['weekday'] = df['start_time'].dt.weekday
df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

mu = df['duration_minutes'].mean()
sigma = df['duration_minutes'].std()

df = df[df['duration_minutes'].between(mu - 5 * sigma, mu + 5 * sigma)]

passholder_encoder = LabelEncoder()
df['passholder_type'] = passholder_encoder.fit_transform(df['passholder_type'])

station_encoder = LabelEncoder()
df['start_station'] = station_encoder.fit_transform(df['start_station'].astype(str))
df['end_station'] = station_encoder.fit_transform(df['end_station'].astype(str))

if "Unknown" not in station_encoder.classes_:
    station_encoder.classes_ = np.append(station_encoder.classes_, "Unknown")

numerical_features = ['hour', 'weekday', 'duration_minutes']
categorical_features = ['start_station', 'end_station']
X_numerical = df[numerical_features]
X_categorical = df[categorical_features]
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df['passholder_type']

imputer = SimpleImputer(strategy='mean')
imputer.fit(df[numerical_features])
df[numerical_features] = imputer.transform(df[numerical_features])

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)
X_scaled = np.hstack((X_numerical_scaled, X_categorical.values))

model = RandomForestClassifier(n_estimators=200, random_state=42, min_samples_split=2,max_depth= None, min_samples_leaf= 2)
model.fit(X_scaled, y)

test_file_path = "test_set.csv"
df_test = pd.read_csv(test_file_path, dtype=str, low_memory=False)

df_test['start_time'] = pd.to_datetime(df_test['start_time'], errors='coerce')
df_test['end_time'] = pd.to_datetime(df_test['end_time'], errors='coerce')
df_test['hour'] = df_test['start_time'].dt.hour
df_test['weekday'] = df_test['start_time'].dt.weekday
df_test['duration_minutes'] = (df_test['end_time'] - df_test['start_time']).dt.total_seconds() / 60


def safe_transform(encoder, value):
    if pd.isna(value):
        value = "Unknown"
    else:
        value = str(value).strip()
    if value not in encoder.classes_:
        value = "Unknown"
    return encoder.transform([value])[0]

df_test['start_station'] = df_test['start_station'].apply(lambda x: safe_transform(station_encoder, x))
df_test['end_station'] = df_test['end_station'].apply(lambda x: safe_transform(station_encoder, x))

df_test[numerical_features] = imputer.transform(df_test[numerical_features])
df_test['start_station'] = df_test['start_station'].fillna("Unknown")
df_test['end_station'] = df_test['end_station'].fillna("Unknown")

X_test_numerical = df_test[numerical_features]
X_test_categorical = df_test[categorical_features]

X_test_numerical_scaled = scaler.transform(X_test_numerical)
X_test_scaled = np.hstack((X_test_numerical_scaled, np.array(X_test_categorical)))

y_pred = model.predict(X_test_scaled)
df_test['passholder_type'] = passholder_encoder.inverse_transform(y_pred)

y_train_pred = model.predict(X_scaled)
importances = model.feature_importances_
feature_names = numerical_features + categorical_features

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importancia de la Variable")
plt.ylabel("Variables")
plt.title("Importancia de Variables en Random Forest")
plt.show()

df_results = df_test[['trip_id', 'passholder_type']]

df_results.to_csv("predictions.csv", index=False)