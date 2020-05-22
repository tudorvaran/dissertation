set -e
python compute_membership.py
set +e

if [ "$1" = "debug" ] ; then
  export SHOW_DEBUG_IMAGES=true
fi
python manage.py runserver localhost:12020