#!/bin/bash

usage() {
    echo "Usage: $0 [-h] [-s] [-f svg|png] filename.dbn"
    echo "  -h : Print help"
    echo "  -s : Show the image file after conversion"
    echo "  -f format : Choose output format (svg or png). Default is png."
    exit 1
}

# check for VARNA
if [ ! -e /opt/VARNA/VARNAv3-93.jar ]
then
    echo "Error: VARNA not found, cannot generate image. Exiting."
    exit 1
fi

SHOW_IMG=false
IMG_FORMAT="png"
DBN_FILE=""

# process args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h) usage ;;
        -s) SHOW_IMG=true
            shift
            ;;
        -f)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "Error: Missing format after -f. Use 'svg' or 'png'."
                usage
            fi
            if [[ "$2" != "svg" && "$2" != "png" ]]; then
                echo "Error: Invalid image format '$2'. Use 'svg' or 'png'."
                usage
            fi
            IMG_FORMAT="$2"
            shift 2
            ;;
        *.dbn)
            if [[ -n "$DBN_FILE" ]]; then
                echo "Error: Multiple .dbn files provided."
                usage
            fi
            DBN_FILE="$1"
            shift
            ;;
        *)
            echo "Error: Invalid argument '$1'"
            usage
            ;;
    esac
done

if [[ -z "$DBN_FILE" ]]; then
    echo "Error: No .dbn file provided."
    usage
fi

IMG_FILE="${DBN_FILE%.dbn}.${IMG_FORMAT}"

VARNAcmd() {
    java -cp /opt/VARNA/VARNAv3-93.jar fr.orsay.lri.varna.applications.VARNAcmd \
                                            "$@" \
                                            -background "#FFFFFF" \
                                            -bp '#000000' \
                                            -baseInner '#cccccc' \
                                            -canvasX 600 -canvasY 300 \
                                            -resolution 15.0 \
                                            -aspectRatio 3.0 \
                                            -algorithm naview
#                                            -algorithm naview, line, circular

}

VARNAcmd -i "$DBN_FILE" -o "$IMG_FILE"

if $SHOW_IMG; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$IMG_FILE"
    elif command -v open >/dev/null 2>&1; then
        open "$IMG_FILE"
    else
        echo "Error: No suitable program found to open image file."
        exit 1
    fi
fi
