__kernel 
void graphics(__global float3* pixels, 
            __constant float3* satelliteIdentifier,
            __constant float2* satellitePosition, 
            float infinity, 
            int satelliteCount, 
            float satelliteRadius, 
            int width)
{
    
    // Row wise ordering
    const int pixelX = get_global_id(0);
    const int pixelY = get_global_id(1);

    // This color is used for coloring the pixel (RGB)
    float3 renderColor = (float3)(0.f, 0.f, 0.f);

    // Find closest satellite
    float shortestDistance = infinity;

    float weights = 0.f;
    int hitsSatellite = 0;

    // First Graphics satellite loop: Find the closest satellite.
    for(int i = 0; i < satelliteCount; ++i){
        float2 difference = (float2)(pixelX - satellitePosition[i].s0, pixelY - satellitePosition[i].s1);
        float distance = sqrt(difference.s0 * difference.s0 + difference.s1 * difference.s1);

        if(distance < satelliteRadius) {
            renderColor.s0 = 1.0f;
            renderColor.s1 = 1.0f;
            renderColor.s2 = 1.0f;
            hitsSatellite = 1;
            break;
        } else {
            float weight = 1.0f / (distance*distance*distance*distance);
            weights = weights + weight;
            if(distance < shortestDistance){
                shortestDistance = distance;
                renderColor = satelliteIdentifier[i];
            }
        }
    }

    // Second graphics loop: Calculate the color based on distance to every satellite.
    if (!hitsSatellite) {
        for(int i = 0; i < satelliteCount; ++i){
            float2 difference = (float2)(pixelX - satellitePosition[i].s0, pixelY - satellitePosition[i].s1);
            float dist2 = difference.s0 * difference.s0 + difference.s1 * difference.s1;
            float weight = 1.0f/(dist2* dist2);

            renderColor.s0 += (satelliteIdentifier[i].s0 * weight /weights) * 3.0f;

            renderColor.s1 += (satelliteIdentifier[i].s1 * weight / weights) * 3.0f;

            renderColor.s2 += (satelliteIdentifier[i].s2 * weight / weights) * 3.0f;
        }
    }

    pixels[width*pixelY + pixelX] = renderColor;
}
