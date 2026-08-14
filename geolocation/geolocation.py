import math


def calculate_target_location(
    drone_lat,
    drone_lon,
    altitude,
    image_width,
    image_height,
    target_x,
    target_y,
    horizontal_fov=90.0,
    vertical_fov=60.0,
):
    """
    Estimate target latitude and longitude from a drone image.

    This is a simplified software prototype.

    Parameters:
        drone_lat       : Drone latitude in degrees
        drone_lon       : Drone longitude in degrees
        altitude        : Drone altitude above ground in meters
        image_width     : Image width in pixels
        image_height    : Image height in pixels
        target_x        : Target center X coordinate in pixels
        target_y        : Target center Y coordinate in pixels
        horizontal_fov  : Camera horizontal field of view in degrees
        vertical_fov    : Camera vertical field of view in degrees

    Returns:
        target_lat, target_lon
    """

    # Convert target position from pixels
    # to angular offset from image center.
    center_x = image_width / 2
    center_y = image_height / 2

    angle_x = (
        (target_x - center_x)
        / image_width
    ) * horizontal_fov

    angle_y = (
        (target_y - center_y)
        / image_height
    ) * vertical_fov

    # Convert angles to radians
    angle_x_rad = math.radians(angle_x)
    angle_y_rad = math.radians(angle_y)

    # Estimate ground displacement in meters.
    east_offset = altitude * math.tan(angle_x_rad)

    north_offset = altitude * math.tan(angle_y_rad)

    # Approximate meters per degree.
    meters_per_degree_lat = 111320

    # Longitude scale depends on latitude.
    meters_per_degree_lon = (
        111320 * math.cos(math.radians(drone_lat))
    )

    target_lat = (
        drone_lat
        + north_offset / meters_per_degree_lat
    )

    target_lon = (
        drone_lon
        + east_offset / meters_per_degree_lon
    )

    return target_lat, target_lon


if __name__ == "__main__":

    # Simulated drone information
    drone_lat = 12.9716
    drone_lon = 77.5946
    altitude = 100.0

    # Simulated camera/image
    image_width = 1280
    image_height = 720

    # Simulated target center
    target_x = 700
    target_y = 350

    latitude, longitude = calculate_target_location(
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        altitude=altitude,
        image_width=image_width,
        image_height=image_height,
        target_x=target_x,
        target_y=target_y,
    )

    print("Drone location:")
    print(f"Latitude : {drone_lat}")
    print(f"Longitude: {drone_lon}")

    print("\nEstimated target location:")
    print(f"Latitude : {latitude:.6f}")
    print(f"Longitude: {longitude:.6f}")