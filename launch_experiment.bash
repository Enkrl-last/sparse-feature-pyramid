#!/bin/bash

export api=`cat <<EOF
api { 
    # Михаил Куренков's workspace
    web_server: https://app.community.clear.ml
    api_server: https://api.community.clear.ml
    files_server: https://files.community.clear.ml
    credentials {
        "access_key" = "OUFP16UH5H99H71SZWIZ"
        "secret_key" = "uAGvwA5NryE6YeiQOzE3yh3mW1OziHlOhTIXuj0ZDgmPNv6NkY"
    }
}
EOF
`
echo "$api" > ~/clearml.conf

python main.py
