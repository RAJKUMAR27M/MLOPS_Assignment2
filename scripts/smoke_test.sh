#!/bin/bash
# Smoke test script in Bash for CI integration

API_URL=${1:-"http://localhost:8000"}
EXIT_CODE=0

echo "Running Bash smoke tests against $API_URL..."

# Test 1: Health Check
echo -n "Test 1: Health Check... "
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" $API_URL/health)
HTTP_STATUS=$(echo "${HEALTH_RESPONSE}" | tail -c 4)
BODY=$(echo "${HEALTH_RESPONSE}" | head -c -4)

if [ "$HTTP_STATUS" -eq 200 ]; then
  if command -v jq &> /dev/null; then
    STATUS=$(echo $BODY | jq -r .status)
    if [ "$STATUS" == "healthy" ]; then
      echo -e "\e[32mPASS\e[0m"
    else
      echo -e "\e[31mFAIL\e[0m (Unexpected status: $STATUS)"
      EXIT_CODE=1
    fi
  else
    echo -e "\e[32mPASS\e[0m (No jq, assuming healthy body: $BODY)"
  fi
else
  echo -e "\e[31mFAIL\e[0m (HTTP $HTTP_STATUS)"
  EXIT_CODE=1
fi

# Generate dummy image for predict test using dd
DUMMY_IMG="dummy_test_image.jpg"
dd if=/dev/urandom of=$DUMMY_IMG bs=1K count=10 2>/dev/null

# Test 2: Predict
echo -n "Test 2: Predict... "
PREDICT_RESPONSE=$(curl -s -w "%{http_code}" -X POST -F "file=@$DUMMY_IMG" $API_URL/predict)
HTTP_STATUS=$(echo "${PREDICT_RESPONSE}" | tail -c 4)
BODY=$(echo "${PREDICT_RESPONSE}" | head -c -4)

if [ "$HTTP_STATUS" -eq 200 ]; then
  if command -v jq &> /dev/null; then
    PREDICTION=$(echo $BODY | jq -r .prediction)
    if [[ "$PREDICTION" == "cat" || "$PREDICTION" == "dog" ]]; then
       echo -e "\e[32mPASS\e[0m (Prediction: $PREDICTION)"
    else
       echo -e "\e[31mFAIL\e[0m (Unexpected prediction: $PREDICTION)"
       EXIT_CODE=1
    fi
  else
    echo -e "\e[32mPASS\e[0m (No jq, body: $BODY)"
  fi
else
  echo -e "\e[31mFAIL\e[0m (HTTP $HTTP_STATUS)"
  EXIT_CODE=1
fi

rm -f $DUMMY_IMG

# Test 3: Metrics
echo -n "Test 3: Metrics... "
METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/metrics)
if [ "$METRICS_STATUS" -eq 200 ]; then
  echo -e "\e[32mPASS\e[0m"
else
  echo -e "\e[31mFAIL\e[0m (HTTP $METRICS_STATUS)"
  EXIT_CODE=1
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  echo -e "\e[32mALL TESTS PASSED\e[0m"
else
  echo -e "\e[31mSOME TESTS FAILED\e[0m"
fi

exit $EXIT_CODE
