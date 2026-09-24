if [ -f "../Population/.data/C2021_SECCOES_11A.zip" ] ; then
    echo "Census file already exists. Skipping download."
else
    echo "Downloading census data..."
    mkdir -p ../Population/.data
    wget -O ../Population/.data/C2021_SECCOES_11A.zip https://mapas.ine.pt/download/filesGPG/2021Seccoes/nuts3/C2021_SECCOES_11A.zip
fi

if [ -f "../Population/.data/C2021_SECCOES_11A.gpkg" ] ; then
    echo "Census CSV file already exists. Skipping extraction."
else
    echo "Extracting census data..."
    unzip ../Population/.data/C2021_SECCOES_11A.zip -d ../Population/.data/
fi

if [ -f "../Population/.data/C2021_SECCOES_11A_TRANSFORMED.gpkg" ] ; then
    echo "Transformed census CSV file already exists. Skipping transformation."
else
    echo "Transforming census data..."
    python ../Population/oporto/DataCreator.py --expand-census ../Population/.data/C2021_SECCOES_11A.gpkg
fi

if [ ! -f "../PhysicalNetwork/.tmp/amp.osm" ] ; then
    echo "Generating physical network data..."
    cd ../PhysicalNetwork
    python generate_network.py
    cd ../Simulation
fi

mkdir -p ../Population/.data/IMOB2017
touch ../Population/.data/IMOB2017/needed.txt

touch ../Population/.data/IMOB2017/TBL_alojamento_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_alojamento_despesa_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_alojamento_veiculos_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_alojamento_rendimentos_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_individuos_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_tipo_de_passe_AMP.csv.missing
touch ../Population/.data/IMOB2017/TBL_viagens_AMP.csv.missing

echo ".data/IMOB2017/TBL_alojamento_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_alojamento_despesa_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_alojamento_veiculos_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_alojamento_rendimentos_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_individuos_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_tipo_de_passe_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt
echo ".data/IMOB2017/TBL_viagens_AMP.csv\n" >> ../Population/.data/IMOB2017/needed.txt

if [ -f "../Population/.data/places.csv" ] ; then
    echo "Places data already exists. Skipping generation."
else
    echo "Generating places data..."
    python ../Population/oporto/DataCreator.py --generate-places ../PhysicalNetwork/.tmp/amp.osm --output ../Population/.data/places.csv
fi

echo ""
echo "Public data for Porto has been successfully loaded and processed to Population/.data\n"

echo "\033[1mUnfortunately, the IMOB data is not publicly available, so it cannot be included in this repository.\033[0m"
echo "\033[1mPlease refer to the README for instructions on how to obtain and paste it in the Population/.data folder.\033[0m"