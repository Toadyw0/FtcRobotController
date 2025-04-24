import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
// imports

public class ObjectTrackingPipeline extends OpenCvPipeline {
    private Mat hsvImage = new Mat(); // Stores HSV-converted frame
    private List<Point> previousCenters = new ArrayList<>(); // Movement path
    private Point trackedPoint = null; // Last object's center
    private boolean objectLocked = false; // Tracks if an object is locked

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsvImage, Imgproc.COLOR_RGB2HSV); // Convert RGB to HSV

        // Detect red objects
        Mat maskRedLower = detectColor(hsvImage, 0, 15, 100, 255, 80, 255);
        Mat maskRedUpper = detectColor(hsvImage, 165, 180, 100, 255, 80, 255);
        Mat maskRed = new Mat();
        Core.addWeighted(maskRedLower, 1.0, maskRedUpper, 1.0, 0.0, maskRed);

        // Track object movement consistently
        trackLockedObject(maskRed, input);

        return input;
    }

    private Mat detectColor(Mat hsv, int hMin, int hMax, int sMin, int sMax, int vMin, int vMax) {
        Mat mask = new Mat();
        Core.inRange(hsv, new Scalar(hMin, sMin, vMin), new Scalar(hMax, sMax, vMax), mask);
        return mask;
    }

    private void trackLockedObject(Mat mask, Mat input) {
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Point newTrackedPoint = null;
        double maxArea = 0;

        // Iterates through each element in MatOfPoint to determine which object is "most red"
        for (MatOfPoint contour : contours) {
            Rect box = Imgproc.boundingRect(contour);
            double area = box.area();

            if (area > maxArea && area > 150 && area < 5000) {
                maxArea = area;
                newTrackedPoint = new Point(box.x + box.width / 2, box.y + box.height / 2);
            }
        }

        // If an object is found and no object is locked, lock onto it
        if (!objectLocked && newTrackedPoint != null) {
            trackedPoint = newTrackedPoint;
            objectLocked = true;
        }
        // If tracking an object, check if it's still detected
        else if (objectLocked) {
            if (newTrackedPoint != null && distanceBetween(trackedPoint, newTrackedPoint) < 20) {
                trackedPoint = newTrackedPoint; // Update position only if nearby
            } else {
                objectLocked = false; // Unlock if object disappears
            }
        }

        // Store movement path only if tracking an object
        if (objectLocked && trackedPoint != null) {
            previousCenters.add(trackedPoint);
            if (previousCenters.size() > 50) {
                previousCenters.subList(0, previousCenters.size() - 50).clear();
            }

            // Draw movement path
            for (int i = 1; i < previousCenters.size(); i++) {
                Imgproc.line(input, previousCenters.get(i - 1), previousCenters.get(i), new Scalar(0, 255, 0), 2);
            }

            // Highlight tracked position
            Imgproc.circle(input, trackedPoint, 6, new Scalar(255, 255, 255), -1);
        }
    }

    private double distanceBetween(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }
}