<template>
  <div class="demo_result">
    <div class="contrast_section">
      <div
        class="contrast_image_wrapper shadow"
        @mouseover="changeImage('in')"
        @mouseleave="changeImage('out')"
      >
        <div class="contrast_image card">
          <img class="image" :src="image.input" alt="input image" />
          <div class="image_info">
            <div class="image_name">original image</div>
            <div class="image_size">{{ data.size.input }} bytes</div>
          </div>
        </div>
        <div class="contrast_image card" :class="{ hidden }">
          <img class="image" :src="contrastImage.src" :alt="contrastImage.name" />
          <div class="image_info">
            <div class="image_name">{{ contrastImage.name }} image</div>
            <div class="image_size">{{ contrastImage.size }} bytes</div>
          </div>
        </div>
      </div>
      <div class="contrast_info card shadow">
        <div class="info_line clickable" @click="select('output')">
          <div class="image_name">
            <span class="selector" :class="{ selected: selectedImageName == 'output' }"
              >Our method</span
            >
          </div>
          <div class="image_info">
            <div>{{ data.eval.tex_bpp.toFixed(3) }} bpp</div>
            <div>{{ data.size.tex }} bytes (texture size)</div>
            <div>{{ data.size.feat }} bytes (feature size)</div>
            <div>PSNR {{ data.eval.fic_psnr.toFixed(6) }}</div>
            <div>SSIM {{ data.eval.fic_ssim.toFixed(6) }}</div>
          </div>
        </div>
        <div class="divider"></div>
        <div class="info_line clickable" @click="select('jpeg')">
          <div class="image_name">
            <span class="selector" :class="{ selected: selectedImageName == 'jpeg' }">JPEG</span>
          </div>
          <div class="image_info">
            <div>{{ data.eval.jpeg_bpp.toFixed(3) }} bpp</div>
            <div>{{ data.size.jpeg }} bytes</div>
            <div>PSNR {{ data.eval.jpeg_psnr.toFixed(6) }}</div>
            <div>SSIM {{ data.eval.jpeg_ssim.toFixed(6) }}</div>
          </div>
        </div>
      </div>
    </div>
    <div class="card_wrapper" v-viewer="{ filter: excludeIcon }">
      <!-- TODO 提供下载/查看原图功能 -->
      <ImageCard class="card clickable" :src="image.input" name="input"></ImageCard>
      <ImageCard class="card clickable" :src="image.feat" name="feature"></ImageCard>
      <ImageCard class="card clickable" :src="image.output" name="output"></ImageCard>
      <ImageCard class="card clickable" :src="image.jpeg" name="jpeg"></ImageCard>
      <ImageCard class="card clickable" :src="image.tex" name="texture"></ImageCard>
      <ImageCard class="card clickable" :src="image.tex_decoded" name="decoded texture"></ImageCard>
      <ImageCard
        class="card clickable"
        name="compress data"
        src="/assets/archive.png"
        is-icon
        @click.native="downloadFic"
      ></ImageCard>
    </div>
  </div>
</template>

<script>
import ImageCard from '@/components/ImageCard.vue';

export default {
  props: {
    data: {
      type: Object,
      required: true,
    },
  },
  data() {
    return {
      selectedImageName: 'output',
      hidden: false,
    };
  },
  computed: {
    image() {
      return this.data.image;
    },
    contrastImage() {
      return {
        name: this.selectedImageName,
        src: this.image[this.selectedImageName],
        size: this.data.size[this.selectedImageName],
      };
    },
  },
  mounted() {
    const cardWrapper = this.$el.querySelector('.card_wrapper');
    cardWrapper.addEventListener('ready', () => {
      this.viewer = cardWrapper.$viewer;
    });
  },
  methods: {
    changeImage(type) {
      this.hidden = type === 'in';
    },
    select(name) {
      this.selectedImageName = name;
    },
    excludeIcon(e) {
      return ![...e.classList].includes('icon');
    },
    downloadFic() {
      window.open(this.data.data);
    },
  },
  components: {
    ImageCard,
  },
};
</script>

<style lang="scss" scoped>
.demo_result {
  $margin: 20px;
  margin: 0 auto;

  .contrast_section {
    display: grid;
    grid-template-columns: 38.2% 61.8%;
    margin: $margin 0;

    .contrast_image_wrapper {
      max-width: 100%;
      max-height: 100%;
      margin-right: $margin;

      position: relative;

      .contrast_image {
        transition: opacity var(--duration);
        width: 100%;

        .image {
          width: 100%;
          height: 100%;
        }

        &.hidden {
          opacity: 0;
        }

        &:nth-of-type(2) {
          position: absolute;
          top: 0;
          left: 0;
        }
      }

      .image_info {
        display: flex;
        justify-content: space-between;
        margin: 12px;
      }
    }

    .contrast_info {
      margin-left: $margin;
      padding: 50px 60px;

      display: flex;
      flex-direction: column;
      justify-content: space-around;

      .info_line {
        display: flex;
      }

      .selector {
        flex: none;
        position: relative;
        display: flex;
        align-items: center;

        &:before {
          content: '';
          height: 8px;
          width: 8px;
          margin-right: 14px;
          border: 1px solid var(--border);
          box-sizing: border-box;
          border-radius: 50%;
          transition: all var(--duration);
        }

        &.selected:before {
          height: 12px;
          width: 12px;
          border: none;
          background: var(--primary);
        }
      }

      .image_name {
        color: var(--text);
        font-size: 22px;
        line-height: 1.2;
        position: relative;
        flex: 1.4;
      }

      .image_info {
        font-size: 16px;
        line-height: 1.5;
        color: var(--text2);
        flex: 2;
      }

      .divider {
        border-top: var(--border) solid 1px;
      }
    }
  }

  .card_wrapper {
    padding: 40px 0;
    display: grid;
    grid-template-columns: repeat(4, 22.5%);
    gap: 3.33%;
  }
}
</style>
