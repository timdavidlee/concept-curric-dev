# Project Results and Notes

For all the below projects that were tested, cross-validation of 5 was used, resulting in 5 different scores

### Airline Satisfaction

Kaggle dataset: [https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

Features:

```python
    "gender",
    "type_of_travel",
    "class", 
    "age",
    "flight_distance",
    "inflight_wifi_service",
    "departure_arrival_time_convenient",
    "ease_of_online_booking",
    "gate_location",
    "food_and_drink",
    "seat_comfort",
    "inflight_entertainment",
    "on_board_service",
```


How to run

```sh
python -m ink_curriculum.projects.airline_satisfaction
```

For predicting a `Loyal Customer` vs. Not

```
log_reg AUC: [0.928 0.921 0.925 0.921 0.92 ]
tree AUC: [0.997 0.997 0.997 0.996 0.996]
```

### Hotel Reservations

Kaggle Dataset: [https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

Features:

```python
"type_of_meal_plan",
"room_type_reserved",
"market_segment_type",
"arrival_year",
"arrival_month", "arrival_date",
"no_of_adults",
"no_of_children",
"no_of_weekend_nights",
"no_of_week_nights",
"required_car_parking_space",
"repeated_guest",
"no_of_previous_cancellations",
"no_of_previous_bookings_not_canceled",
"avg_price_per_room",
"no_of_special_requests",

```


```sh
python -m ink_curriculum.projects.hotel_reservations
```

From predicting if they will cancel or not, AUC scores:

```sh
log_reg [0.831 0.791 0.789 0.785 0.802]
tree [0.918 0.911 0.913 0.914 0.916]
```

From prediction what the room rate is, given the features:

```sh
log_reg [-23.27  -22.691 -22.571 -23.182 -23.366]
tree [-18.661 -17.558 -17.48  -17.85  -18.632]
```

### ecommerce shipping data

Features:

```python
"warehouse_block",
"mode_of_shipment",
"cost_of_the_product",
"prior_purchases",
"discount_offered",
"weight_in_gms"
```

Kaggle Dataset: [https://www.kaggle.com/datasets/prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)

```sh
python -m ink_curriculum.projects.ecommerce_shipping
```

Predicting if the shipment will be delayed (Y/N)

```
log_reg [0.953 0.953 0.62  0.486 0.488]
tree [0.992 0.994 0.665 0.503 0.522]
```

### Website text classification

Kaggle Dataset: [https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization](https://www.kaggle.com/datasets/bpmtips/websiteiabcategorization)

```sh
python -m ink_curriculum.projects.iab_text
```

Making this a NLP problem, classifying the dataset into a one vs. rest model for specific topics:

Features:

```python
"size",
"image_id",
"domain",
"category",
"title",
"description", # used for source text
"keywords",
```

```
for topic: web_design_development
0.0370463348156649% | 1,033.0

log_reg [0.798 0.819 0.82  0.816 0.843]
tree [0.774 0.798 0.784 0.793 0.806]
pretrained+logreg [0.837 0.836 0.845 0.846 0.879]
```