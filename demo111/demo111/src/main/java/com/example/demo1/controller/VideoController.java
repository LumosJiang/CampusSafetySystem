package com.example.demo1.controller;

import com.example.demo1.entity.Danger;
import com.example.demo1.entity.Result;
import com.example.demo1.entity.Video;
import com.example.demo1.mapper.VideoMapper;
import com.example.demo1.service.VideoService;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Tag(name = "视频控制器", description = "描述：管理视频信息")
@RestController
@RequestMapping("/video")
public class VideoController {
    @Autowired
    private VideoService videoService;

    @Operation(summary = "所有视频")
    @GetMapping("/all")
    public Result allVideo() {
        List<Video> videoList = videoService.allVideo();
        return Result.success(videoList);
    }
    @GetMapping("/alldanger")
    public Result allDanger() {
        List<Danger> dangerList = videoService.allDanger();
        return Result.success(dangerList);
    }
    @Operation(summary = "添加视频")
    @PostMapping("/addVideo")
    public Result addVideo(@RequestBody Video video) {
        int video_id = video.getVideo_id();
        String path = video.getPath();

        // 获取当前时间
        LocalDateTime now = LocalDateTime.now();
        // 定义日期时间格式
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        String date = now.format(formatter);
        videoService.addVideo(video_id, path, date);
        return Result.success();
    }
    @Operation(summary = "通过视频名拿到路径")
    @GetMapping("/replay")
    public Result getPathByName(String name) {
        String path = videoService.getPathByName(name);
        return Result.success(path);
    }
    @Operation(summary = "通过id拿到视频详细信息")
    @GetMapping("/detail")
    public Result getDetail(int video_id) {
        return Result.success(videoService.getDetail(video_id));
    }

    @Operation(summary = "删除视频")
    @PostMapping("/delete")
    public Result delete(int video_id) {
        videoService.delete(video_id);
        return Result.success();
    }
}
