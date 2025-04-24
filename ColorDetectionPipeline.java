import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.List;
// imports


public class ColorDetectionPipeline extends OpenCvPipeline { // "extends OpenCvPipeline" allows for the program to process image/camera
    private Mat hsvImage = new Mat(); // stores new matrix called hsvImage

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsvImage, Imgproc.COLOR_RGB2HSV); // converts the RGB format to HSV

        // Masks are created for all 3 colors; range values are entered (H - hue, S - saturation, V - value (darkness))
        Mat maskRed = detectColor(hsvImage, 0, 10, 120, 255, 100, 255);
        Mat maskBlue = detectColor(hsvImage, 100, 140, 100, 255, 50, 255);
        Mat maskYellow = detectColor(hsvImage, 20, 40, 120, 255, 120, 255);


        // Draws the color coded borders around the objects and outputs the color
        detectObjects(maskRed, input, "red", new Scalar(255, 0, 0));
        detectObjects(maskBlue, input, "blue", new Scalar(0, 0, 255));
        detectObjects(maskYellow, input, "yellow", new Scalar(255, 165, 0));

        return input;
    }

    /* defines detectColor() function,
    Isolates certain colors (red, blue, or yellow) to determine whether or not they are considered said color
    hMin/hMax = hue min/max
    sMin/sMax = saturation min/max
    vMin/vMax = value min/max
     */
    private Mat detectColor(Mat hsv, int hMin, int hMax, int sMin, int sMax, int vMin, int vMax) {
        Mat mask = new Mat();
        Core.inRange(hsv, new Scalar(hMin, sMin, vMin), new Scalar(hMax, sMax, vMax), mask); // detected colors are set aside differently than non-detected colors
        return mask;
    }

    private void detectObjects(Mat mask, Mat input, String label, Scalar boxColor) {
        List<MatOfPoint> contours = new java.util.ArrayList<>();
        Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        /* finds the outlines of the detected colors
        Imgproc.RETR_EXTERNAL returns only the external boundaries
        Imgproc.CHAIN_APPROX_SIMPLE ensures lines aren't jagged
         */

        for (MatOfPoint contour : contours) { // loops through all the boundaries of the list MatOfPoint
            Rect box = Imgproc.boundingRect(contour); // creates the box

            // Reduces noise by setting restrictions: 100 < area or area < 3500; width > 130 or height > 130
            if (box.area() < 100 || box.area() > 3500) continue;
            if (box.width > 130 || box.height > 130) continue;

            // Draws the boxes for us to view
            Imgproc.rectangle(input, box.tl(), box.br(), boxColor, 1);

            // Draws the labels
            Imgproc.putText(input, label, new Point(box.x + 5, box.y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.25, new Scalar(255, 255, 255), 1);
        }
    }
}